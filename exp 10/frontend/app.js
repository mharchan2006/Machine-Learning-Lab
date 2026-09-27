// Frontend application logic for MelanomaClassifier

let currentSelectedFile = null;
let currentSampleType = null;

document.addEventListener('DOMContentLoaded', () => {
    initDropZone();
    initSampleButtons();
    initActionButtons();
    loadHistory();
    loadEvaluationMetrics();
    drawDefaultFeatureChart();
});

// --- Drop Zone & File Selection ---
function initDropZone() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#3b82f6';
        dropZone.style.backgroundColor = '#101b33';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = '#2a374a';
        dropZone.style.backgroundColor = '#0d1527';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#2a374a';
        dropZone.style.backgroundColor = '#0d1527';
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

function handleFileSelect(file) {
    currentSelectedFile = file;
    currentSampleType = null;

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            showInputPreview(e.target.result, file.name, (file.size / 1024).toFixed(1) + ' KB', `${img.width} × ${img.height} px`);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function initSampleButtons() {
    document.getElementById('btn-sample-melanoma').addEventListener('click', () => {
        currentSelectedFile = null;
        currentSampleType = 'melanoma';
        showInputPreview(
            '/uploads/melanoma_dermoscopy_sample.jpg',
            'melanoma_dermoscopy_sample.jpg',
            '342.6 KB',
            '380 × 380 px'
        );
        // Direct auto-trigger for benchmark sample
        triggerProcessing();
    });

    document.getElementById('btn-sample-benign').addEventListener('click', () => {
        currentSelectedFile = null;
        currentSampleType = 'benign';
        showInputPreview(
            '/uploads/benign_nevus_sample.jpg',
            'benign_nevus_sample.jpg',
            '318.4 KB',
            '380 × 380 px'
        );
        // Direct auto-trigger for benchmark sample
        triggerProcessing();
    });
}

function showInputPreview(src, filename, size, resolution) {
    document.getElementById('preview-raw-img').src = src;
    document.getElementById('meta-filename').textContent = filename;
    document.getElementById('meta-size').textContent = size;
    document.getElementById('meta-res').textContent = resolution;
    document.getElementById('input-preview-card').style.display = 'flex';
    document.getElementById('alert-step-1').style.display = 'flex';
    document.getElementById('alert-step-1-text').textContent = 'Image uploaded successfully. Ready to run pipeline...';
    
    // Set step 1 active in tracker
    setStepTracker(1);
    document.getElementById('badge-step-1').textContent = 'READY';
    document.getElementById('badge-step-1').className = 'status-badge ready-badge';
}

function initActionButtons() {
    document.getElementById('btn-process-classify').addEventListener('click', triggerProcessing);
    document.getElementById('btn-retrain').addEventListener('click', handleRetrain);
    document.getElementById('btn-view-curves').addEventListener('click', () => {
        const curvesImg = document.getElementById('training-curves-img');
        curvesImg.scrollIntoView({ behavior: 'smooth' });
    });
    document.getElementById('btn-reevaluate').addEventListener('click', loadEvaluationMetrics);
    document.getElementById('btn-refresh-history').addEventListener('click', loadHistory);
    document.getElementById('btn-clear-history').addEventListener('click', clearHistory);
}

// --- Main Pipeline Execution ---
async function triggerProcessing() {
    const btn = document.getElementById('btn-process-classify');
    btn.textContent = '⏳ Processing Dermoscopy Pipeline...';
    btn.disabled = true;

    try {
        const formData = new FormData();
        if (currentSampleType) {
            formData.append('sample', currentSampleType);
        } else if (currentSelectedFile) {
            formData.append('image', currentSelectedFile);
        } else {
            alert('Please select or upload an image first.');
            btn.textContent = '🚀 Process & Classify Image';
            btn.disabled = false;
            return;
        }

        const resp = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        const data = await resp.json();

        if (data.error) {
            alert('Error processing image: ' + data.error);
            btn.textContent = '🚀 Process & Classify Image';
            btn.disabled = false;
            return;
        }

        // 1. Update Step 1 Status
        document.getElementById('badge-step-1').textContent = 'DONE ✓';
        document.getElementById('badge-step-1').className = 'status-badge done-badge';
        document.getElementById('alert-step-1-text').textContent = 'Image uploaded successfully. Running pipeline...';

        // 2. Update Step 2: Preprocessing
        document.getElementById('badge-step-2').textContent = 'DONE ✓';
        document.getElementById('badge-step-2').className = 'status-badge done-badge';
        document.getElementById('prep-orig-img').src = data.original_image_url;
        document.getElementById('prep-enhanced-img').src = data.preprocessed_image_url;
        document.getElementById('prep-orig-dim').textContent = data.resolution;
        document.getElementById('alert-step-2').style.display = 'flex';

        // 3. Update Step 3: Feature Extraction
        document.getElementById('badge-step-3').textContent = 'DONE ✓';
        document.getElementById('badge-step-3').className = 'status-badge done-badge';
        const sampleBox = document.getElementById('vector-sample-display') || document.getElementById('vector-sample-box');
        if (sampleBox) {
            sampleBox.textContent = `[${data.feature_sample.join(', ')}, ...]`;
        }
        drawFeatureChart(data.feature_vector_all);
        document.getElementById('alert-step-3').style.display = 'flex';

        // 4. Update Step 5: Classification Result
        document.getElementById('badge-step-5').textContent = 'DONE ✓';
        document.getElementById('badge-step-5').className = 'status-badge done-badge';

        const diagCard = document.getElementById('diagnostic-card');
        const headline = document.getElementById('diagnostic-headline');
        const confSpan = document.getElementById('diag-confidence');
        const noteBox = document.getElementById('clinical-note-box');
        const noteText = document.getElementById('clinical-note-text');

        confSpan.textContent = data.confidence + '%';
        document.getElementById('prob-bar-melanoma').style.width = data.melanoma_prob + '%';
        document.getElementById('prob-num-melanoma').textContent = data.melanoma_prob + '%';
        document.getElementById('prob-bar-benign').style.width = data.benign_prob + '%';
        document.getElementById('prob-num-benign').textContent = data.benign_prob + '%';

        if (data.is_malignant) {
            diagCard.className = 'diagnostic-card melanoma-style';
            headline.textContent = 'MALIGNANT MELANOMA DETECTED';
            noteText.textContent = data.clinical_advice;
        } else {
            diagCard.className = 'diagnostic-card benign-style';
            headline.textContent = 'NORMAL / BENIGN NEVUS (HEALTHY)';
            noteText.textContent = data.clinical_advice;
        }

        document.getElementById('alert-step-5').style.display = 'flex';
        document.getElementById('alert-pred-name').textContent = data.prediction;
        document.getElementById('alert-pred-conf').textContent = data.confidence + '%';

        // Update Tracker to 100%
        setStepTracker(7);

        // Refresh History Table
        loadHistory();

        // Smooth scroll to prediction
        diagCard.scrollIntoView({ behavior: 'smooth', block: 'center' });

    } catch (err) {
        console.error('Processing error:', err);
        alert('Diagnostic process error: ' + (err.message || 'Please check terminal logs.'));
    } finally {
        btn.textContent = '🚀 Process & Classify Image';
        btn.disabled = false;
    }
}

// --- Step Progress Tracker Updater ---
function setStepTracker(activeUpToStep) {
    const totalSteps = 7;
    for (let i = 1; i <= totalSteps; i++) {
        const node = document.getElementById(`tracker-step-i` ? `tracker-step-${i}` : null);
        if (node) {
            if (i <= activeUpToStep) {
                node.classList.add('completed');
                node.classList.add('active');
            } else {
                node.classList.remove('completed');
                node.classList.remove('active');
            }
        }
    }
    const percent = Math.min(100, Math.round((activeUpToStep / totalSteps) * 100));
    document.getElementById('tracker-fill').style.width = `${percent}%`;
}

// --- Feature Extraction Canvas Visualizer ---
function drawFeatureChart(featureValues) {
    const canvas = document.getElementById('feature-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Draw baseline
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 20);
    ctx.lineTo(width, height - 20);
    ctx.stroke();

    const n = featureValues.length;
    const barWidth = (width - 40) / n;
    const maxVal = Math.max(1.0, ...featureValues);

    for (let i = 0; i < n; i++) {
        const val = Math.max(0.01, featureValues[i]);
        const barHeight = Math.min(height - 35, (val / maxVal) * (height - 40));
        const x = 20 + i * barWidth;
        const y = height - 20 - barHeight;

        // Visual gradient matching reference
        if (i < 32) {
            ctx.fillStyle = '#3b4d6e'; // R histogram
        } else if (i < 64) {
            ctx.fillStyle = '#0284c7'; // G histogram
        } else if (i < 96) {
            ctx.fillStyle = '#10b981'; // B histogram
        } else {
            ctx.fillStyle = '#84cc16'; // Gradients & Texture
        }

        ctx.fillRect(x, y, Math.max(2, barWidth - 1), barHeight);
    }
}

function drawDefaultFeatureChart() {
    // Generate default sample distribution for initial render
    const sample = [];
    for (let i = 0; i < 128; i++) {
        if (i < 96) {
            sample.push(Math.abs(Math.sin(i * 0.15)) * 0.8 + 0.1);
        } else {
            sample.push(Math.abs(Math.cos(i * 0.2)) * 0.9 + 0.05);
        }
    }
    drawFeatureChart(sample);
}

// --- Neural Network Retraining ---
async function handleRetrain() {
    const btn = document.getElementById('btn-retrain');
    btn.textContent = '🔄 Training (15 Epochs)...';
    btn.disabled = true;

    try {
        const resp = await fetch('/api/train', { method: 'POST' });
        const data = await resp.json();

        if (data.error) {
            alert('Training error: ' + data.error);
            return;
        }

        // Render live epoch logs
        const terminal = document.getElementById('epoch-log-terminal');
        terminal.innerHTML = '';
        data.logs.forEach(log => {
            const line = document.createElement('div');
            line.className = 'log-line' + (log.epoch === 15 ? ' highlight-log' : '');
            line.textContent = `Epoch ${log.epoch}: Acc ${log.train_acc}% | Loss ${log.loss.toFixed(4)} | Val Acc ${log.val_acc}%`;
            terminal.appendChild(line);
        });

        // Refresh training curves image
        document.getElementById('training-curves-img').src = data.curve_url + '?t=' + Date.now();
        document.getElementById('alert-train-success').style.display = 'flex';

        // Re-evaluate
        await loadEvaluationMetrics();

    } catch (err) {
        console.error(err);
        alert('Failed to connect to training API.');
    } finally {
        btn.textContent = '🔄 Retrain Neural Network';
        btn.disabled = false;
    }
}

// --- Performance Evaluation ---
async function loadEvaluationMetrics() {
    const btn = document.getElementById('btn-reevaluate');
    btn.textContent = '⏳ Evaluating...';
    btn.disabled = true;

    try {
        const resp = await fetch('/api/evaluate');
        const data = await resp.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        document.getElementById('metric-acc').textContent = data.accuracy;
        document.getElementById('metric-prec').textContent = data.precision;
        document.getElementById('metric-rec').textContent = data.recall;
        document.getElementById('metric-f1').textContent = data.f1_score;

        // Refresh CM Image
        document.getElementById('cm-image').src = data.cm_image_url + '?t=' + Date.now();

    } catch (err) {
        console.error('Failed to load metrics:', err);
    } finally {
        btn.textContent = '📊 Re-evaluate Model on Test Set';
        btn.disabled = false;
    }
}

// --- SQLite Prediction History ---
async function loadHistory() {
    try {
        const resp = await fetch('/api/history');
        const data = await resp.json();
        const tbody = document.getElementById('history-table-body');
        tbody.innerHTML = '';

        if (!data.history || data.history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="empty-table-msg">No prediction history records in database yet.</td></tr>';
            return;
        }

        data.history.forEach(item => {
            const tr = document.createElement('tr');
            const isMel = item.prediction.toLowerCase().includes('melanoma');
            const badgeClass = isMel ? 'pill-melanoma' : 'pill-benign';

            tr.innerHTML = `
                <td><code>${escapeHtml(item.filename)}</code></td>
                <td><span class="pill-badge ${badgeClass}">${item.prediction}</span></td>
                <td><strong>${Number(item.confidence).toFixed(1)}%</strong></td>
                <td>${item.created_at}</td>
            `;
            tbody.appendChild(tr);
        });

    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

async function clearHistory() {
    if (!confirm('Are you sure you want to clear all prediction history?')) return;
    try {
        await fetch('/api/history/clear', { method: 'POST' });
        loadHistory();
    } catch (err) {
        console.error('Failed to clear history:', err);
    }
}

function escapeHtml(text) {
    if (!text) return '';
    return text.replace(/[&<>"']/g, function(m) {
        return {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        }[m];
    });
}
