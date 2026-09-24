use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("path/to/your_model.onnx")
        .out_dir("path/to/burn_models")
        .run_from_cli();
}
