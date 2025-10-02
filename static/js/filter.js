(function(){
  // DOM elements
  const fileInput = document.getElementById('fileInput');
  const fileNameEl = document.getElementById('fileName');
  const originalPreview = document.getElementById('originalPreview');

  const groupSelect = document.getElementById('groupSelect');
  const opSelect = document.getElementById('opSelect');
  const paramsPanel = document.getElementById('paramsPanel');

  const addStepBtn = document.getElementById('addStepBtn');
  const previewBtn = document.getElementById('previewBtn');
  const clearStepsBtn = document.getElementById('clearStepsBtn');
  const stepsList = document.getElementById('stepsList');

  const resultImg = document.getElementById('resultImg');
  const downloadBtn = document.getElementById('downloadBtn');

  const gradXImg = document.getElementById('gradXImg');
  const gradYImg = document.getElementById('gradYImg');
  const magImg = document.getElementById('magImg');
  const threshImg = document.getElementById('threshImg');
  const lapImg = document.getElementById('lapImg');
  const singleRunBtn = document.getElementById('singleRunBtn');

  let imageDataUrl = null;
  let steps = [];

  const opsByGroup = {
    grayscale: [
      { value: 'grayscale', label: 'Convert to Grayscale', params: [] },
    ],
    point: [
      { value: 'point_negative', label: 'Âm bản (Negative)', params: [] },
      { value: 'point_log', label: 'Log Transform', params: [] },
      { value: 'point_power', label: 'Gamma (Power)', params: [
        { key: 'gamma', label: 'Gamma', type: 'range', min: 0.1, max: 5.0, step: 0.1, value: 0.5 },
      ] },
      { value: 'point_hist_eq', label: 'Cân bằng Histogram', params: [] },
      { value: 'point_piecewise', label: 'Đoạn thẳng (Piecewise Linear)', params: [
        { key: 'r1', label: 'r1', type: 'range', min: 0, max: 255, step: 1, value: 100 },
        { key: 's1', label: 's1', type: 'range', min: 0, max: 255, step: 1, value: 0 },
        { key: 'r2', label: 'r2', type: 'range', min: 0, max: 255, step: 1, value: 200 },
        { key: 's2', label: 's2', type: 'range', min: 0, max: 255, step: 1, value: 255 },
      ] },
      { value: 'point_clahe', label: 'CLAHE', params: [
        { key: 'clip_limit', label: 'Clip Limit', type: 'range', min: 1.0, max: 10.0, step: 0.1, value: 2.0 },
        { key: 'tiles', label: 'Tile Grid Size', type: 'range', min: 2, max: 16, step: 1, value: 8 },
      ] },
    ],
    noise: [
      { value: 'noise_gaussian', label: 'Gaussian Noise', params: [
        { key: 'mean', label: 'Mean', type: 'range', min: -50, max: 50, step: 1, value: 0 },
        { key: 'var', label: 'Variance', type: 'range', min: 0, max: 100, step: 1, value: 10 },
      ]},
      { value: 'noise_sp', label: 'Salt & Pepper Noise', params: [
        { key: 'amount', label: 'Amount', type: 'range', min: 0.0, max: 0.2, step: 0.005, value: 0.02 },
        { key: 's_vs_p', label: 'Salt vs Pepper', type: 'range', min: 0.0, max: 1.0, step: 0.01, value: 0.5 },
      ]},
    ],
    blur: [
      { value: 'blur_mean', label: 'Mean Blur', params: [ { key: 'ksize', label: 'Kernel Size', type: 'odd-range', min: 1, max: 31, step: 2, value: 3 } ] },
      { value: 'blur_gaussian', label: 'Gaussian Blur', params: [
        { key: 'ksize', label: 'Kernel Size', type: 'odd-range', min: 1, max: 31, step: 2, value: 3 },
        { key: 'sigma', label: 'Sigma', type: 'range', min: 0.1, max: 10.0, step: 0.1, value: 1.0 },
      ] },
      { value: 'blur_median', label: 'Median Blur', params: [ { key: 'ksize', label: 'Kernel Size', type: 'odd-range', min: 1, max: 31, step: 2, value: 3 } ] },
      { value: 'blur_bilateral', label: 'Bilateral Filter', params: [
        { key: 'diameter', label: 'Diameter', type: 'odd-range', min: 1, max: 31, step: 2, value: 9 },
        { key: 'sigma_color', label: 'Sigma Color', type: 'range', min: 1, max: 250, step: 1, value: 75 },
        { key: 'sigma_space', label: 'Sigma Space', type: 'range', min: 1, max: 250, step: 1, value: 75 },
      ] },
    ],
    sharpen: [
      { value: 'sharpen_laplacian', label: 'Laplacian Sharpen', params: [
        { key: 'ksize', label: 'Kernel Size', type: 'odd-range', min: 1, max: 7, step: 2, value: 3 },
        { key: 'alpha', label: 'Alpha', type: 'range', min: 0.0, max: 2.0, step: 0.1, value: 1.0 },
      ] },
      { value: 'sharpen_unsharp', label: 'Unsharp Mask', params: [
        { key: 'ksize', label: 'Kernel Size', type: 'odd-range', min: 1, max: 31, step: 2, value: 5 },
        { key: 'sigma', label: 'Sigma', type: 'range', min: 0.1, max: 10.0, step: 0.1, value: 1.0 },
        { key: 'amount', label: 'Amount', type: 'range', min: 0.0, max: 3.0, step: 0.1, value: 1.0 },
        { key: 'threshold', label: 'Threshold', type: 'range', min: 0, max: 255, step: 1, value: 0 },
      ] },
    ],
    edge: [
      { value: 'edge_sobel', label: 'Sobel', params: [ { key: 'threshold', label: 'Threshold', type: 'range', min: 0, max: 255, step: 1, value: 100 } ] },
      { value: 'edge_prewitt', label: 'Prewitt', params: [ { key: 'threshold', label: 'Threshold', type: 'range', min: 0, max: 255, step: 1, value: 100 } ] },
      { value: 'edge_laplacian', label: 'Laplacian', params: [
        { key: 'ksize', label: 'Kernel Size', type: 'odd-range', min: 1, max: 7, step: 2, value: 3 },
        { key: 'threshold', label: 'Threshold', type: 'range', min: 0, max: 255, step: 1, value: 100 },
      ] },
      { value: 'edge_canny', label: 'Canny', params: [
        { key: 't1', label: 'Lower Threshold', type: 'range', min: 0, max: 255, step: 1, value: 100 },
        { key: 't2', label: 'Upper Threshold', type: 'range', min: 0, max: 255, step: 1, value: 200 },
      ] },
    ],
  };

  function getEngine() {
    const radios = document.querySelectorAll('input[name="engine"]');
    for (const r of radios) if (r.checked) return r.value;
    return 'cv2';
  }

  function populateOps() {
    const group = groupSelect.value;
    const list = opsByGroup[group] || [];
    opSelect.innerHTML = '';
    list.forEach((op, idx) => {
      const opt = document.createElement('option');
      opt.value = op.value;
      opt.textContent = op.label;
      if (idx === 0) opt.selected = true;
      opSelect.appendChild(opt);
    });
    renderParams();
  }

  function createSliderControl(p) {
    // Ensure odd value for odd-range
    let val = p.value;
    if (p.type === 'odd-range') {
      val = Math.max(p.min ?? 1, Math.floor(val));
      if (val % 2 === 0) val += 1;
    }
    const wrapper = document.createElement('div');
    wrapper.className = 'bg-[#10253f] rounded p-3';
    const id = `param_${p.key}`;
    const min = p.min ?? 0;
    const max = p.max ?? 100;
    const step = p.step ?? 1;

    wrapper.innerHTML = `
      <div class="flex items-center justify-between mb-1">
        <label for="${id}">${p.label}</label>
        <span id="${id}_val" class="text-white/80 text-sm">${val}</span>
      </div>
      <input type="range" id="${id}" data-key="${p.key}" class="w-full" min="${min}" max="${max}" step="${step}" value="${val}" />
    `;

    // enforce odd for odd-range on input
    const input = wrapper.querySelector('input');
    const valSpan = wrapper.querySelector(`#${id}_val`);
    input.addEventListener('input', (e) => {
      let v = (step % 1 !== 0) ? parseFloat(e.target.value) : parseInt(e.target.value, 10);
      if (p.type === 'odd-range') {
        v = Math.max(min, Math.round(v));
        if (v % 2 === 0) v += 1;
        e.target.value = v;
      }
      valSpan.textContent = v;
    });

    return wrapper;
  }

  function renderParams() {
    paramsPanel.innerHTML = '';
    const group = groupSelect.value;
    const list = opsByGroup[group] || [];
    const current = list.find(x => x.value === opSelect.value);
    if (!current || !current.params) return;

    current.params.forEach(p => {
      if (p.type === 'range' || p.type === 'odd-range') {
        paramsPanel.appendChild(createSliderControl(p));
      } else {
        const wrap = document.createElement('div');
        wrap.className = 'bg-[#10253f] rounded p-3';
        wrap.innerHTML = `
          <label class="block mb-1">${p.label}</label>
          <input type="number" step="${p.step || '1'}" value="${p.value}" data-key="${p.key}" class="w-full rounded p-2 text-black" />
        `;
        paramsPanel.appendChild(wrap);
      }
    });
  }

  function getCurrentParams() {
    const inputs = paramsPanel.querySelectorAll('input');
    const params = {};
    inputs.forEach(inp => {
      const key = inp.getAttribute('data-key');
      let val = inp.value;
      // slider returns string; parse to number
      if (inp.type === 'range' || inp.type === 'number') {
        const num = Number(val);
        if (!Number.isNaN(num)) val = num;
      }
      params[key] = val;
    });
    // enforce odd for any known odd-range keys by name
    ['ksize', 'diameter'].forEach(k => {
      if (params[k] !== undefined) {
        let v = Math.max(1, Math.round(params[k]));
        if (v % 2 === 0) v += 1;
        params[k] = v;
      }
    });
    return params;
  }

  function findOpLabel(opValue) {
    for (const group in opsByGroup) {
      const entry = (opsByGroup[group] || []).find(o => o.value === opValue);
      if (entry) return entry.label || opValue;
    }
    return opValue;
  }

  function paramSummary(params) {
    const keys = Object.keys(params);
    if (!keys.length) return '(no params)';
    return keys.map(k => `${k}=${params[k]}`).join(', ');
  }

  function renderStepsList() {
    stepsList.innerHTML = '';
    if (!steps.length) {
      const empty = document.createElement('div');
      empty.className = 'text-white/80 text-sm';
      empty.textContent = 'Chưa có thao tác nào. Hãy thêm thao tác bằng nút phía trên.';
      stepsList.appendChild(empty);
      return;
    }

    steps.forEach((step, idx) => {
      const row = document.createElement('div');
      row.className = 'flex items-center justify-between bg-[#10253f] rounded p-3';
      const left = document.createElement('div');
      left.innerHTML = `<div class="font-semibold">${idx+1}. ${findOpLabel(step.op)}</div>
                        <div class="text-sm text-white/70">${paramSummary(step.params || {})}</div>`;
      const right = document.createElement('div');
      right.className = 'flex items-center gap-2';
      const upBtn = document.createElement('button');
      upBtn.className = 'px-2 py-1 rounded bg-white/10 hover:bg-white/20';
      upBtn.innerHTML = '<i class="fa fa-arrow-up"></i>';
      upBtn.disabled = idx === 0;
      upBtn.addEventListener('click', () => {
        if (idx > 0) {
          const tmp = steps[idx-1];
          steps[idx-1] = steps[idx];
          steps[idx] = tmp;
          renderStepsList();
        }
      });
      const downBtn = document.createElement('button');
      downBtn.className = 'px-2 py-1 rounded bg-white/10 hover:bg-white/20';
      downBtn.innerHTML = '<i class="fa fa-arrow-down"></i>';
      downBtn.disabled = idx === steps.length - 1;
      downBtn.addEventListener('click', () => {
        if (idx < steps.length - 1) {
          const tmp = steps[idx+1];
          steps[idx+1] = steps[idx];
          steps[idx] = tmp;
          renderStepsList();
        }
      });
      const delBtn = document.createElement('button');
      delBtn.className = 'px-2 py-1 rounded bg-red-500/70 hover:bg-red-500';
      delBtn.innerHTML = '<i class="fa fa-trash"></i>';
      delBtn.addEventListener('click', () => {
        steps.splice(idx, 1);
        renderStepsList();
      });

      right.appendChild(upBtn);
      right.appendChild(downBtn);
      right.appendChild(delBtn);
      row.appendChild(left);
      row.appendChild(right);
      stepsList.appendChild(row);
    });
  }

  async function previewPipeline() {
    if (!imageDataUrl) {
      alert('Vui lòng chọn ảnh trước.');
      return;
    }
    if (!steps.length) {
      alert('Chưa có thao tác nào trong pipeline.');
      return;
    }
    const engine = getEngine();
    try {
      previewBtn.disabled = true;
      previewBtn.innerHTML = '<i class="fa fa-spinner fa-spin mr-1"></i> Đang xem trước...';
      const res = await fetch('/api/vision/pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: imageDataUrl, engine, steps })
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Xem trước thất bại');
      resultImg.src = data.output;
    } catch (e) {
      console.error(e);
      alert('Lỗi: ' + e.message);
    } finally {
      previewBtn.disabled = false;
      previewBtn.innerHTML = '<i class="fa fa-eye mr-1"></i> Xem trước pipeline';
    }
  }

  async function singleRunCurrent() {
    if (!imageDataUrl) {
      alert('Vui lòng chọn ảnh trước.');
      return;
    }
    const engine = getEngine();
    const op = opSelect.value;
    const params = getCurrentParams();

    gradXImg.src = '';
    gradYImg.src = '';
    magImg.src = '';
    threshImg.src = '';
    lapImg.src = '';

    try {
      singleRunBtn.disabled = true;
      singleRunBtn.innerHTML = 'Đang chạy...';
      const res = await fetch('/api/vision/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: imageDataUrl, op, engine, params })
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Xử lý thất bại');
      resultImg.src = data.output || '';
      gradXImg.src = data.grad_x || '';
      gradYImg.src = data.grad_y || '';
      magImg.src = data.magnitude || '';
      threshImg.src = data.threshold || '';
      lapImg.src = data.laplacian || '';
    } catch (e) {
      console.error(e);
      alert('Lỗi: ' + e.message);
    } finally {
      singleRunBtn.disabled = false;
      singleRunBtn.innerHTML = 'Chạy đơn thao tác hiện tại';
    }
  }

  function addCurrentStep() {
    const op = opSelect.value;
    const params = getCurrentParams();
    steps.push({ op, params });
    renderStepsList();
  }

  function clearSteps() {
    steps = [];
    renderStepsList();
  }

  function downloadResult() {
    if (!resultImg.src) {
      alert('Chưa có ảnh kết quả để tải.');
      return;
    }
    const a = document.createElement('a');
    a.href = resultImg.src;
    a.download = 'result.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  // Event bindings
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    fileNameEl.textContent = file.name;
    const reader = new FileReader();
    reader.onload = () => {
      imageDataUrl = reader.result;
      originalPreview.src = imageDataUrl;
      resultImg.src = '';
      gradXImg.src = gradYImg.src = magImg.src = threshImg.src = lapImg.src = '';
    };
    reader.readAsDataURL(file);
  });

  groupSelect.addEventListener('change', populateOps);
  opSelect.addEventListener('change', renderParams);

  addStepBtn.addEventListener('click', addCurrentStep);
  previewBtn.addEventListener('click', previewPipeline);
  clearStepsBtn.addEventListener('click', clearSteps);
  singleRunBtn.addEventListener('click', singleRunCurrent);
  downloadBtn.addEventListener('click', downloadResult);

  // init
  populateOps();
  renderStepsList();
})();
