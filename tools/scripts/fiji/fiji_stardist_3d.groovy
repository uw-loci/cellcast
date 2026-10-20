#@ Img (label="Input image:", autofill=false) image
#@ String (label="Model:", choices={"Demo 3D"}, style="listBox") model
#@ Double (label="Percentile minimum:", value=1.0, style="format:0.00", min=0.00, max=100.00, stepSize=1.0, description="minimum percentile value for normalization") pmin
#@ Double (label="Percentile maximum:", value=99.8, sytle="format:0.00", min=0.00, max=1.00, stepSize=1.0, description="maximum percentile value for normalization") pmax
#@ Double (label="Probability threshold:", value=0.479, style="format:0.000", min=0.000, max=1.000, stepSize=0.1, description="Polygon probability threshold") prob_threshold
#@ Double (label="NMS threshold:", value=0.3, style="format:0.00", min=0.00, max=1.00, stepSize=0.1, description="Non-Maximum Suppression threshold") nms_threshold
#@ Boolean (label="GPU", value=true, description="Set True for GPU inference via WebGPU, False for CPU inference") gpu
#@output Img labels

import org.apposed.appose.Appose

cellcastScript = """
def share_as_ndarray(img):
    ""\"Copies a NumPy array into a same-sized newly allocated block of shared memory""\"
    from appose import NDArray
    shared = NDArray(str(img.dtype), img.shape)
    shared.ndarray()[:] = img
    return shared

from cellcast.models import StarDist3D
if model == "Demo 3D":
	sd = StarDist3D.init_fluo(gpu=gpu)
	labels = sd.predict_fluo(
	    image.ndarray(),
	    pmin,
	    pmax,
	    prob_threshold,
	    nms_threshold,
	)
share_as_ndarray(labels)
"""

println("== BUILDING ENVIRONMENT ==")
env = Appose.uv().include("cellcast==0.3.2").name("cellcast").logDebug().build()
println("Environment build complete: ${env.base()}")

// Conversion functions: ImgLib2 Img <-> Appose NDArray
imgToAppose = { img ->
	ndArray = net.imglib2.appose.ShmImg.copyOf(image).ndArray()
	println("Copied image into shared memory: ${ndArray.shape()} DType{${ndArray.dType()}}")
	return ndArray
}
apposeToImg = { ndarray ->
	net.imglib2.appose.NDArrays.asArrayImg(ndarray)
}
copyImg = { img ->
	// Note: We use PlanarImg because the original ImageJ likes them best.
	copy = new net.imglib2.img.planar.PlanarImgFactory(img.getType()).create(img.dimensionsAsLongArray())
	net.imglib2.util.ImgUtil.copy(img, copy)
	return copy
}

// Run the script as an Appose task
println("== STARTING PYTHON SERVICE ==")
try (python = env.python().init("import numpy")) {
	inputs = [
		"image": imgToAppose(image),
		"pmin": pmin,
		"pmax": pmax,
		"prob_threshold": prob_threshold,
		"nms_threshold": nms_threshold,
		"gpu": gpu,
		"model": model,
	]
	task = python.task(cellcastScript, inputs)
		.listen { if (it.message) println("[CELLCAST] ${it.message}") }
		.waitFor()

	println("TASK FINISHED: ${task.status}")
	if (task.error) println(task.error)
	labels = copyImg(apposeToImg(task.result()))
	// Without the copyImg, imglib2-ij fails to wrap such ArrayImgs to ImagePlus,
	// due to ImageProcessorUtils expecting a backing Java primitive array type.
	//
	// pixels = labels.update( null ).getCurrentStorageArray()
	// ^ Pixels is a DirectShortBufferU here -- makes sense; it's shared memory
	//
	// But then... how the heck does the unseg_fiji plugin work?!
	// So for the moment, we just copy it into a non-shm Img. :'-(	
}
finally {
	println("== TERMINATING PYTHON SERVICE ==")
}
