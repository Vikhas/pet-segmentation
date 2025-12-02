// State
let currentTab = 'cutout';
const files = {
    cutout: null,
    compare: null,
    fg: null,
    bg: null
};

// Elements
const loader = document.getElementById('loader');

// Tab Switching
function switchTab(tab) {
    currentTab = tab;

    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.currentTarget.classList.add('active');

    // Update panels
    document.querySelectorAll('.panel').forEach(panel => panel.classList.remove('active'));
    document.getElementById(`${tab}-panel`).classList.add('active');
}

// Drag and Drop Setup
function setupDragDrop(zoneId, inputId, type) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');

        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0], type);
        }
    });

    input.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0], type);
        }
    });
}

function handleFile(file, type) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }

    files[type] = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewId = type === 'cutout' ? 'cutout-preview' : `${type}-preview`;
        const img = document.getElementById(`${previewId}-img`);
        const container = document.getElementById(`${previewId}-container`);

        img.src = e.target.result;
        container.style.display = 'flex';

        updateButtons();
    };
    reader.readAsDataURL(file);
}

function clearCutout() {
    files.cutout = null;
    document.getElementById('cutout-preview-container').style.display = 'none';
    document.getElementById('cutout-input').value = '';
    document.getElementById('cutout-result').style.display = 'none';
    updateButtons();
}

function clearComposite(type) {
    files[type] = null;
    document.getElementById(`${type}-preview-container`).style.display = 'none';
    document.getElementById(`${type}-input`).value = '';
    document.getElementById('composite-result').style.display = 'none';
    updateButtons();
}

function updateButtons() {
    document.getElementById('cutout-btn').disabled = !files.cutout;
    document.getElementById('composite-btn').disabled = !(files.fg && files.bg);
}

// API Calls
async function generateCutout() {
    if (!files.cutout) return;

    showLoader();
    const formData = new FormData();
    formData.append('image', files.cutout);

    // Get selected model
    const selectedModel = document.querySelector('input[name="cutout-model"]:checked').value;
    formData.append('model', selectedModel);

    try {
        const response = await fetch('/api/cutout', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Processing failed');

        const data = await response.json();

        if (data.error) throw new Error(data.error);

        const resultImg = document.getElementById('cutout-result-img');
        const downloadLink = document.getElementById('cutout-download');

        // Update Process Viz
        const originalPreview = document.getElementById('cutout-preview-img');
        document.getElementById('viz-original').src = originalPreview.src;
        document.getElementById('viz-matte').src = data.alpha_matte;
        document.getElementById('viz-cutout').src = data.image;

        resultImg.src = data.image;
        downloadLink.href = data.image;

        // Update Metrics
        if (data.metrics) {
            document.getElementById('metric-confidence').textContent = (data.metrics.confidence * 100).toFixed(1) + '%';
            document.getElementById('metric-iou').textContent = data.metrics.benchmark_iou.toFixed(4);
            document.getElementById('metric-dice').textContent = data.metrics.benchmark_dice.toFixed(4);
            document.getElementById('metric-mae').textContent = data.metrics.benchmark_mae.toFixed(4);
        }

        document.getElementById('cutout-result').style.display = 'block';

    } catch (error) {
        alert('Error generating cutout: ' + error.message);
    } finally {
        hideLoader();
    }
}

async function generateComposite() {
    if (!files.fg || !files.bg) return;

    showLoader();
    const formData = new FormData();
    formData.append('foreground', files.fg);
    formData.append('background', files.bg);

    try {
        const response = await fetch('/api/composite', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Processing failed');

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        const resultImg = document.getElementById('composite-result-img');
        const downloadLink = document.getElementById('composite-download');

        resultImg.src = url;
        downloadLink.href = url;

        document.getElementById('composite-result').style.display = 'block';

    } catch (error) {
        alert('Error generating composite: ' + error.message);
    } finally {
        hideLoader();
    }
}

async function startCompositeEditor() {
    if (!files.fg || !files.bg) return;

    showLoader();

    try {
        // Step 1: Get Cutout for Foreground
        const formData = new FormData();
        formData.append('image', files.fg);

        // Get selected model
        const selectedModel = document.querySelector('input[name="composite-model"]:checked').value;
        formData.append('model', selectedModel);

        const response = await fetch('/api/cutout', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Processing failed');
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        // Step 2: Setup Editor
        const bgImg = document.getElementById('editor-bg');
        const fgImg = document.getElementById('editor-fg');

        // Load BG
        const bgReader = new FileReader();
        bgReader.onload = (e) => {
            bgImg.src = e.target.result;

            // Load FG (Cutout)
            fgImg.src = data.image;

            // Reset FG position
            fgImg.style.top = '50%';
            fgImg.style.left = '50%';
            fgImg.style.transform = 'translate(-50%, -50%) scale(1)';
            currentScale = 1;

            // Show Editor
            document.getElementById('composite-editor').style.display = 'block';
            document.getElementById('composite-btn').style.display = 'none';

            hideLoader();
        };
        bgReader.readAsDataURL(files.bg);

    } catch (error) {
        hideLoader();
        alert('Error preparing editor: ' + error.message);
    }
}

// Editor Interaction Logic
let isDragging = false;
let currentX;
let currentY;
let initialX;
let initialY;
let xOffset = 0;
let yOffset = 0;
let currentScale = 1;

const fgElement = document.getElementById('editor-fg');
const container = document.getElementById('canvas-container');

// Dragging
container.addEventListener('mousedown', dragStart);
container.addEventListener('mouseup', dragEnd);
container.addEventListener('mousemove', drag);
container.addEventListener('mouseleave', dragEnd);

// Touch support
container.addEventListener('touchstart', dragStart, { passive: false });
container.addEventListener('touchend', dragEnd);
container.addEventListener('touchmove', drag, { passive: false });

// Scaling (Wheel)
container.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY * -0.001;
    currentScale = Math.min(Math.max(.1, currentScale + delta), 3);
    updateTransform();
});

function dragStart(e) {
    if (e.type === "touchstart") {
        initialX = e.touches[0].clientX - xOffset;
        initialY = e.touches[0].clientY - yOffset;
    } else {
        initialX = e.clientX - xOffset;
        initialY = e.clientY - yOffset;
    }

    if (e.target === fgElement) {
        isDragging = true;
    }
}

function dragEnd(e) {
    initialX = currentX;
    initialY = currentY;
    isDragging = false;
}

function drag(e) {
    if (isDragging) {
        e.preventDefault();

        if (e.type === "touchmove") {
            currentX = e.touches[0].clientX - initialX;
            currentY = e.touches[0].clientY - initialY;
        } else {
            currentX = e.clientX - initialX;
            currentY = e.clientY - initialY;
        }

        xOffset = currentX;
        yOffset = currentY;

        updateTransform();
    }
}

function updateTransform() {
    // We use translate3d for better performance
    // Note: We combine the center-positioning logic with the drag offset
    fgElement.style.transform = `translate(calc(-50% + ${currentX}px), calc(-50% + ${currentY}px)) scale(${currentScale})`;
}

function closeEditor() {
    document.getElementById('composite-editor').style.display = 'none';
    document.getElementById('composite-btn').style.display = 'block';
    // Reset state
    xOffset = 0;
    yOffset = 0;
    currentScale = 1;
}

function downloadComposite() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const bgImg = document.getElementById('editor-bg');
    const fgImg = document.getElementById('editor-fg');

    // Set canvas size to match BG image natural size
    canvas.width = bgImg.naturalWidth;
    canvas.height = bgImg.naturalHeight;

    // Draw BG
    ctx.drawImage(bgImg, 0, 0);

    // Calculate FG position and size relative to the canvas
    // This is tricky because the DOM elements are scaled by CSS object-fit

    // 1. Get the displayed dimensions of BG
    const bgRect = bgImg.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    // Calculate the scale factor between displayed BG and actual BG
    // object-fit: contain means the image is fitted within the container
    const displayRatio = bgRect.width / bgImg.naturalWidth;

    // 2. Get FG position relative to BG center
    // Our CSS transform is relative to center: translate(-50% + x, -50% + y)
    // So xOffset and yOffset are the distance from the center in SCREEN PIXELS

    // Convert screen pixel offsets to canvas pixel offsets
    const canvasXOffset = xOffset / displayRatio;
    const canvasYOffset = yOffset / displayRatio;

    // 3. Calculate FG dimensions on canvas
    const fgNaturalWidth = fgImg.naturalWidth;
    const fgNaturalHeight = fgImg.naturalHeight;

    // Initial max-width: 50% constraint in CSS needs to be accounted for?
    // Actually, let's just use the computed width/height from the DOM to be safe about CSS constraints
    const fgRect = fgImg.getBoundingClientRect();

    // The scale is already applied to fgRect by getBoundingClientRect
    // But we need the "unrotated/unscaled" base size to apply our currentScale manually?
    // No, simpler: 
    // The FG width on screen is fgRect.width.
    // The FG width on canvas should be fgRect.width / displayRatio.

    const fgCanvasWidth = fgRect.width / displayRatio;
    const fgCanvasHeight = fgRect.height / displayRatio;

    // 4. Calculate Position
    // Center of canvas
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    // Top-left position
    const fgX = centerX + canvasXOffset - (fgCanvasWidth / 2);
    const fgY = centerY + canvasYOffset - (fgCanvasHeight / 2);

    // Draw FG
    ctx.drawImage(fgImg, fgX, fgY, fgCanvasWidth, fgCanvasHeight);

    // Download
    const link = document.createElement('a');
    link.download = 'pet_composite.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
}

function showLoader() {
    loader.style.display = 'flex';
}

function hideLoader() {
    loader.style.display = 'none';
}

// Initialize
setupDragDrop('cutout-dropzone', 'cutout-input', 'cutout');
setupDragDrop('fg-dropzone', 'fg-input', 'fg');
setupDragDrop('bg-dropzone', 'bg-input', 'bg');
setupDragDrop('compare-dropzone', 'compare-input', 'compare');

// Comparison Mode
function clearCompare() {
    files.compare = null;
    document.getElementById('compare-preview-container').style.display = 'none';
    document.getElementById('compare-input').value = '';
    document.getElementById('compare-result').style.display = 'none';
    updateButtons();
}

function updateButtons() {
    document.getElementById('cutout-btn').disabled = !files.cutout;
    document.getElementById('composite-btn').disabled = !(files.fg && files.bg);
    if (document.getElementById('compare-btn')) {
        document.getElementById('compare-btn').disabled = !files.compare;
    }
}

async function compareModels() {
    if (!files.compare) return;

    showLoader();
    const formData = new FormData();
    formData.append('image', files.compare);

    try {
        const response = await fetch('/api/compare', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Comparison failed');

        const data = await response.json();

        if (data.error) throw new Error(data.error);

        // Update MODNet results
        document.getElementById('modnet-cutout').src = data.modnet.image;
        document.getElementById('modnet-confidence').textContent = (data.modnet.metrics.confidence * 100).toFixed(1) + '%';
        document.getElementById('modnet-iou').textContent = data.modnet.metrics.benchmark_iou.toFixed(4);
        document.getElementById('modnet-dice').textContent = data.modnet.metrics.benchmark_dice.toFixed(4);
        document.getElementById('modnet-mae').textContent = data.modnet.metrics.benchmark_mae.toFixed(4);

        // Update FBA results (if available)
        if (data.fba) {
            document.getElementById('fba-cutout').src = data.fba.image;
            document.getElementById('fba-confidence').textContent = (data.fba.metrics.confidence * 100).toFixed(1) + '%';
            document.getElementById('fba-iou').textContent = data.fba.metrics.benchmark_iou.toFixed(4);
            document.getElementById('fba-dice').textContent = data.fba.metrics.benchmark_dice.toFixed(4);
            document.getElementById('fba-mae').textContent = data.fba.metrics.benchmark_mae.toFixed(4);

            // Determine winner
            const modnetScore = data.modnet.metrics.benchmark_dice;
            const fbaScore = data.fba.metrics.benchmark_dice;
            const winnerBanner = document.getElementById('winner-banner');
            const winnerText = document.getElementById('winner-text');

            if (modnetScore > fbaScore) {
                winnerText.textContent = `MODNet performs better (Dice: ${modnetScore.toFixed(4)} vs ${fbaScore.toFixed(4)})`;
            } else if (fbaScore > modnetScore) {
                winnerText.textContent = `FBA performs better (Dice: ${fbaScore.toFixed(4)} vs ${modnetScore.toFixed(4)})`;
            } else {
                winnerText.textContent = 'Both models perform equally well!';
            }
            winnerBanner.style.display = 'block';
        } else {
            alert('FBA model not available. Showing MODNet results only.');
        }

        document.getElementById('compare-result').style.display = 'block';

    } catch (error) {
        alert('Error comparing models: ' + error.message);
    } finally {
        hideLoader();
    }
}
