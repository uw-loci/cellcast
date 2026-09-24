"""
fix_asymmetric_padding.py
-------------------------
Rewrites ONNX Conv nodes that carry asymmetric explicit `pads` attributes so
that burn-import / onnx-ir (which only supports symmetric padding) can consume
the model.

For each Conv node whose ONNX `pads` vector has at least one dimension where
  begin_pad[i] != end_pad[i]
we insert an explicit `Pad` op (opset-18 style, takes a separate `pads` input
tensor) immediately before the Conv, zero-out the Conv's own `pads` attribute,
and wire them together.  Every other attribute (strides, dilations, group,
kernel_shape, auto_pad) is left untouched.

The result is bit-for-bit identical to the original graph:
  original: x ──► Conv(pads=[a,b,...,c,d,...])
  fixed:    x ──► Pad(pads_tensor=[…0…,a,b,…,c,d,…]) ──► Conv(pads=[0,0,…])

Usage
-----
  python scripts/fix_asymmetric_padding.py

Output: onnx_models/stardist_3d_3d_demo_sym_padded.onnx
"""

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
import pathlib

INPUT_PATH  = pathlib.Path("path/to/your_model.onnx")
OUTPUT_PATH = pathlib.Path("path/to/your_model_fixed.onnx")


def has_asymmetric_pads(node: onnx.NodeProto) -> tuple[bool, list[int]]:
    """Return (is_asymmetric, pads_list) for a Conv node."""
    for attr in node.attribute:
        if attr.name == "pads":
            pads = list(attr.ints)
            ndim = len(pads) // 2
            begins = pads[:ndim]
            ends   = pads[ndim:]
            if begins != ends:
                return True, pads
    return False, []


def split_pads_for_burn(pads: list[int]) -> tuple[list[int], list[int]]:
    """
    `onnx-ir` Pad nodes only allow nonzero values at the last two spatial dims
    (H and W, i.e. positions [ndim-2], [ndim-1], [2*ndim-2], [2*ndim-1] in the
    full NCDHW pads tensor).  D padding must therefore remain in the Conv, which
    only accepts symmetric values.

    Given Conv3d pads [D_b, H_b, W_b, D_e, H_e, W_e], returns:
      conv_pads  – pads to write onto the Conv node (D kept, H/W zeroed)
      pad_pads   – full NCDHW pads tensor for the Pad node (D must be 0)

    Raises ValueError if D padding is asymmetric (cannot be represented).
    """
    assert len(pads) == 6, f"Expected 6-element pads for Conv3d, got {pads}"
    D_b, H_b, W_b, D_e, H_e, W_e = pads

    if D_b != D_e:
        raise ValueError(
            f"D padding is asymmetric ({D_b} vs {D_e}); cannot split for "
            "burn-import. A manual fix is required."
        )

    # Pad node pads for a 5D NCDHW tensor:
    # [N_b, C_b, D_b, H_b, W_b,  N_e, C_e, D_e, H_e, W_e]
    # D must be 0 here; H and W carry the asymmetric values.
    pad_pads  = [0, 0, 0, H_b, W_b,  0, 0, 0, H_e, W_e]

    # Conv keeps only the symmetric D padding; H/W are now zero.
    conv_pads = [D_b, 0, 0, D_e, 0, 0]

    return conv_pads, pad_pads


def make_pad_node(
    conv_name: str,
    input_name: str,
    pad_pads: list[int],
) -> tuple[onnx.NodeProto, onnx.TensorProto, str]:
    """
    Build a constant initializer + Pad node for the given full NCDHW pads.
    """
    pad_initializer_name = f"{conv_name}_asym_pads_const"
    pad_output_name      = f"{conv_name}_asym_pad_out"

    pads_tensor = numpy_helper.from_array(
        np.array(pad_pads, dtype=np.int64),
        name=pad_initializer_name,
    )

    pad_node = helper.make_node(
        op_type="Pad",
        inputs=[input_name, pad_initializer_name],
        outputs=[pad_output_name],
        name=f"{conv_name}_asym_pad",
    )

    return pad_node, pads_tensor, pad_output_name


def fix_model(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = list(graph.initializer)
    fixed_count = 0

    for node in graph.node:
        if node.op_type not in ("Conv", "ConvInteger"):
            new_nodes.append(node)
            continue

        is_asym, pads = has_asymmetric_pads(node)
        if not is_asym:
            new_nodes.append(node)
            continue

        # Split into what Pad handles (H, W) vs what stays in Conv (D symmetric)
        conv_pads, pad_pads = split_pads_for_burn(pads)

        original_input = node.input[0]
        pad_node, pads_tensor, padded_output = make_pad_node(
            conv_name=node.name or f"conv_{id(node)}",
            input_name=original_input,
            pad_pads=pad_pads,
        )

        new_nodes.append(pad_node)
        new_initializers.append(pads_tensor)

        # Rebuild the Conv node: padded input, symmetric (or zero) pads
        new_attrs = []
        for attr in node.attribute:
            if attr.name == "pads":
                new_attrs.append(helper.make_attribute("pads", conv_pads))
            else:
                new_attrs.append(attr)

        new_inputs = [padded_output] + list(node.input[1:])
        new_conv = helper.make_node(
            op_type=node.op_type,
            inputs=new_inputs,
            outputs=list(node.output),
            name=node.name,
        )
        new_conv.attribute.extend(new_attrs)
        if node.doc_string:
            new_conv.doc_string = node.doc_string

        new_nodes.append(new_conv)

        print(
            f"  Fixed '{node.name}': pads {pads}"
            f"\n    → Pad(NCDHW) {pad_pads}"
            f"\n    → Conv pads  {conv_pads}"
        )
        fixed_count += 1

    # Rebuild graph
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=new_initializers,
    )
    # Copy value_info (intermediate tensor type/shape annotations)
    new_graph.value_info.extend(graph.value_info)

    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    if model.doc_string:
        new_model.doc_string = model.doc_string

    if fixed_count == 0:
        print("No asymmetric Conv padding found — model is already compatible.")
    else:
        print(f"\nFixed {fixed_count} Conv node(s) with asymmetric padding.")

    return new_model


def main() -> None:
    print(f"Loading {INPUT_PATH} …")
    model = onnx.load(str(INPUT_PATH))

    print("Scanning for Conv nodes with asymmetric pads …")
    fixed_model = fix_model(model)

    print(f"\nChecking fixed model …")
    onnx.checker.check_model(fixed_model)
    print("Model check passed.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(fixed_model, str(OUTPUT_PATH))
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
