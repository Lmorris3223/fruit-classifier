const classes = ["apple", "banana", "orange"]; 

let session = null;


async function loadModel() {
    try {
        session = await ort.InferenceSession.create("fruit_model.onnx");
        console.log("Model loaded");
        console.log("Inputs:", session.inputNames);
        console.log("Outputs:", session.outputNames);
    } catch (error) {
        console.error("Error loading model:", error);
    }
}

loadModel();


document.getElementById("imageInput").addEventListener("change", async (event) => {
    const file = event.target.files[0];

    if (!session) {
        alert("Model not loaded yet!");
        return;
    }

    const img = document.getElementById("preview");
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
        try {
            const tensor = preprocessImage(img);

            
            const feeds = {};
            feeds[session.inputNames[0]] = tensor;

            const results = await session.run(feeds);

          
            const outputName = session.outputNames[0];
            const output = results[outputName].data;

            console.log("Raw output:", output);

            const prediction = output.indexOf(Math.max(...output));

            document.getElementById("result").innerText =
                "Prediction: " + classes[prediction];

        } catch (error) {
            console.error("Prediction error:", error);
        }
    };
});

function preprocessImage(image) {
    const canvas = document.createElement("canvas");
    canvas.width = 224;
    canvas.height = 224;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0, 224, 224);

    const imageData = ctx.getImageData(0, 0, 224, 224).data;

    const floatData = new Float32Array(3 * 224 * 224);

    for (let i = 0; i < 224 * 224; i++) {
        floatData[i] = imageData[i * 4] / 255;                      
        floatData[i + 224 * 224] = imageData[i * 4 + 1] / 255;      
        floatData[i + 2 * 224 * 224] = imageData[i * 4 + 2] / 255;   
    }

    return new ort.Tensor("float32", floatData, [1, 3, 224, 224]);
}