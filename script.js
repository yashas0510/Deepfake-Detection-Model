document.getElementById('upload-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('file');
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    const resultDiv = document.getElementById('result');

    if (result.error) {
        resultDiv.textContent = `Error: ${result.error}`;
        resultDiv.style.color = 'red';
    } else {
        resultDiv.textContent = `Prediction: ${result.label} (Confidence: ${(result.confidence * 100).toFixed(2)}%)`;
        resultDiv.style.color = 'green';
    }
});
