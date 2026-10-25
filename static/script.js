// static/script.js - Intelligent Prompt Quality Assessment Framework

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('assess-form');
  const promptInput = document.getElementById('prompt-input');
  const counter = document.getElementById('counter');
  const analyzeBtn = document.getElementById('analyze-btn');
  const clearBtn = document.getElementById('clear-btn');
  const importBtn = document.getElementById('import-btn');
  const fileImportInput = document.getElementById('file-import-input');
  const importBadge = document.getElementById('import-badge');
  const multiPromptBar = document.getElementById('multi-prompt-bar');
  const promptSelector = document.getElementById('prompt-selector');
  const errorBanner = document.getElementById('error-banner');
  const importBanner = document.getElementById('import-banner');

  let importedPromptsList = [];

  const resultContainer = document.getElementById('result-container');
  const overallScoreVal = document.getElementById('overall-score-val');
  const categoryBadge = document.getElementById('category-badge');

  const valClarity = document.getElementById('val-clarity');
  const barClarity = document.getElementById('bar-clarity');
  const valSpecificity = document.getElementById('val-specificity');
  const barSpecificity = document.getElementById('bar-specificity');
  const valCompleteness = document.getElementById('val-completeness');
  const barCompleteness = document.getElementById('bar-completeness');
  const valRelevance = document.getElementById('val-relevance');
  const barRelevance = document.getElementById('bar-relevance');

  const suggestionsList = document.getElementById('suggestions-list');
  const improvedPromptText = document.getElementById('improved-prompt-text');
  const copyBtn = document.getElementById('copy-btn');
  const useImprovedBtn = document.getElementById('use-improved-btn');

  // Enterprise Domain Management
  let selectedDomain = 'General';
  const domainTabs = document.querySelectorAll('.domain-tab');
  const selectedDomainDisplay = document.getElementById('selected-domain-display');
  const domainInput = document.getElementById('domain-input');
  const resultDomainVal = document.getElementById('result-domain-val');

  function setDomain(domain) {
    const valid = ['HR', 'Finance', 'Marketing', 'General'];
    selectedDomain = valid.includes(domain) ? domain : 'General';
    domainTabs.forEach(tab => {
      if (tab.getAttribute('data-domain') === selectedDomain) {
        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');
      } else {
        tab.classList.remove('active');
        tab.setAttribute('aria-selected', 'false');
      }
    });
    if (selectedDomainDisplay) {
      selectedDomainDisplay.textContent = selectedDomain;
    }
    if (domainInput) {
      domainInput.value = selectedDomain;
    }
  }

  domainTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const d = tab.getAttribute('data-domain');
      setDomain(d);
    });
  });

  window.fillDomainPrompt = function(domain, text) {
    setDomain(domain);
    fillPrompt(text);
  };

  // Update live character and word counters
  function updateCounter() {
    if (!promptInput || !counter) return;
    const text = promptInput.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    const chars = promptInput.value.length;
    counter.textContent = `${words} words · ${chars} characters`;
  }

  if (promptInput) {
    promptInput.addEventListener('input', updateCounter);
    updateCounter();
  }

  // Clear button handler
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      promptInput.value = '';
      setDomain('General');
      updateCounter();
      hideError();
      hideSuccess();
      if (importBadge) {
        importBadge.textContent = '';
        importBadge.classList.add('hidden');
      }
      if (multiPromptBar) {
        multiPromptBar.classList.add('hidden');
      }
      if (fileImportInput) {
        fileImportInput.value = '';
      }
      importedPromptsList = [];
      if (resultContainer) {
        resultContainer.classList.add('hidden');
      }
      promptInput.focus();
    });
  }

  // Helper to show errors
  function showError(msg) {
    if (importBanner) importBanner.classList.add('hidden');
    if (errorBanner) {
      errorBanner.textContent = msg;
      errorBanner.classList.remove('hidden');
    } else {
      alert(msg);
    }
  }

  function hideError() {
    if (errorBanner) {
      errorBanner.classList.add('hidden');
    }
  }

  function showSuccess(msg) {
    hideError();
    if (importBanner) {
      importBanner.textContent = msg;
      importBanner.classList.remove('hidden');
    }
  }

  function hideSuccess() {
    if (importBanner) {
      importBanner.classList.add('hidden');
    }
  }

  // Helper to parse file content into list of prompts
  function extractPromptsFromText(filename, text) {
    const ext = filename.toLowerCase();
    const prompts = [];

    if (ext.endsWith('.csv')) {
      const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
      if (lines.length > 0) {
        const header = lines[0].split(',').map(c => c.trim().toLowerCase());
        const pIdx = header.indexOf('prompt');
        const startIdx = pIdx !== -1 ? 1 : 0;
        const colIdx = pIdx !== -1 ? pIdx : 0;
        for (let i = startIdx; i < lines.length; i++) {
          const cols = lines[i].split(',');
          if (cols[colIdx] && cols[colIdx].trim()) {
            prompts.push(cols[colIdx].trim().replace(/^["']|["']$/g, ''));
          }
        }
      }
    } else if (ext.endsWith('.json')) {
      const data = JSON.parse(text);
      const list = Array.isArray(data) ? data : (data.prompts || data.data || []);
      list.forEach(item => {
        if (typeof item === 'string' && item.trim()) {
          prompts.push(item.trim());
        } else if (item && typeof item === 'object') {
          const p = item.prompt || item.text || '';
          if (p.trim()) prompts.push(p.trim());
        }
      });
    } else {
      // Treat as plain text
      text.split(/\r?\n/).forEach(line => {
        const trimmed = line.trim();
        if (trimmed) prompts.push(trimmed);
      });
    }
    return prompts;
  }

  // Load imported prompts into the "Enter Prompt for Evaluation" section
  function loadImportedPrompts(filename, prompts) {
    if (!prompts || prompts.length === 0) {
      showError(`No valid prompts found in file "${filename}".`);
      return;
    }

    importedPromptsList = prompts;

    // Put first prompt into textarea without automatically running
    promptInput.value = importedPromptsList[0];
    updateCounter();

    // Show file badge
    if (importBadge) {
      importBadge.textContent = `📁 ${filename} (${prompts.length} prompt${prompts.length > 1 ? 's' : ''})`;
      importBadge.classList.remove('hidden');
    }

    // Configure multi-prompt selector if more than 1 prompt
    if (multiPromptBar && promptSelector) {
      if (prompts.length > 1) {
        promptSelector.innerHTML = '';
        prompts.forEach((p, idx) => {
          const opt = document.createElement('option');
          opt.value = idx;
          const snippet = p.length > 65 ? p.substring(0, 65) + '...' : p;
          opt.textContent = `${idx + 1}. ${snippet}`;
          promptSelector.appendChild(opt);
        });
        multiPromptBar.classList.remove('hidden');

        promptSelector.onchange = () => {
          const selectedIdx = parseInt(promptSelector.value, 10);
          if (importedPromptsList[selectedIdx]) {
            promptInput.value = importedPromptsList[selectedIdx];
            updateCounter();
          }
        };
      } else {
        multiPromptBar.classList.add('hidden');
      }
    }

    // Show proper guidance notification
    showSuccess(
      `✓ File "${filename}" loaded (${prompts.length} prompt${prompts.length > 1 ? 's' : ''}). Review the prompt below and click "Analyze Prompt" to evaluate.`
    );

    // Scroll to prompt textarea and focus
    promptInput.focus();
  }

  // Check if prompts were imported from History page
  const pendingImport = sessionStorage.getItem('imported_prompts');
  if (pendingImport) {
    try {
      const parsed = JSON.parse(pendingImport);
      sessionStorage.removeItem('imported_prompts');
      loadImportedPrompts(parsed.filename || 'imported_file', parsed.prompts || []);
    } catch (e) {
      console.error('Error parsing session import:', e);
    }
  }

  // Handle file import on Assessment page
  if (importBtn && fileImportInput) {
    importBtn.addEventListener('click', () => {
      fileImportInput.click();
    });

    fileImportInput.addEventListener('change', async () => {
      const file = fileImportInput.files[0];
      if (!file) return;

      hideError();
      hideSuccess();

      try {
        const text = await file.text();
        const prompts = extractPromptsFromText(file.name, text);
        loadImportedPrompts(file.name, prompts);
      } catch (err) {
        console.error('Import error:', err);
        showError(`Failed to parse file "${file.name}": ${err.message}`);
      } finally {
        fileImportInput.value = '';
      }
    });
  }


  // Helper to fill prompt from chips
  window.fillPrompt = function (text) {
    if (!promptInput) return;
    promptInput.value = text;
    updateCounter();
    hideError();
    hideSuccess();
    promptInput.focus();
  };


  // Form submission handler
  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      hideError();

      const prompt = promptInput.value.trim();
      if (!prompt) {
        showError('Please enter a prompt to analyze.');
        return;
      }
      if (prompt.length < 3) {
        showError('Prompt is too short. Please provide a meaningful prompt.');
        return;
      }

      // Show loading spinner
      analyzeBtn.disabled = true;
      const btnText = analyzeBtn.querySelector('.btn-text');
      const spinner = analyzeBtn.querySelector('.spinner');
      if (btnText) btnText.textContent = 'Analyzing with ML Model...';
      if (spinner) spinner.classList.remove('hidden');

      try {
        const response = await fetch('/analyze', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({ prompt, domain: selectedDomain })
        });

        const data = await response.json();

        if (!response.ok || data.error) {
          throw new Error(data.error || 'Failed to analyze prompt.');
        }

        renderResult(data);
      } catch (err) {
        console.error('Analysis error:', err);
        showError(err.message || 'An error occurred while connecting to the backend.');
      } finally {
        analyzeBtn.disabled = false;
        if (btnText) btnText.textContent = 'Analyze Prompt';
        if (spinner) spinner.classList.add('hidden');
      }
    });
  }

  // Render assessment result in DOM
  function renderResult(data) {
    if (!resultContainer) return;

    // Show card
    resultContainer.classList.remove('hidden');

    // Display selected domain
    if (resultDomainVal) {
      resultDomainVal.textContent = data.domain || selectedDomain || 'General';
    }

    // Overall Score (as integer)
    overallScoreVal.textContent = Math.round(Number(data.overall_score) || 0);

    // Quality Category Badge
    const category = data.category || 'Average';
    categoryBadge.textContent = category;
    categoryBadge.className = 'badge';
    const slug = category.toLowerCase().replace(/\s+/g, '-');
    categoryBadge.classList.add(`badge-${slug}`);

    // Dimension Scores and Bars
    const clarity = Math.round(data.clarity || 0);
    valClarity.textContent = `${clarity}%`;
    barClarity.style.width = `${clarity}%`;

    const specificity = Math.round(data.specificity || 0);
    valSpecificity.textContent = `${specificity}%`;
    barSpecificity.style.width = `${specificity}%`;

    const completeness = Math.round(data.completeness || 0);
    valCompleteness.textContent = `${completeness}%`;
    barCompleteness.style.width = `${completeness}%`;

    const relevance = Math.round(data.relevance || 0);
    valRelevance.textContent = `${relevance}%`;
    barRelevance.style.width = `${relevance}%`;

    // Suggestions List
    suggestionsList.innerHTML = '';
    const sugs = data.suggestions || [];
    if (sugs.length === 0) {
      const li = document.createElement('li');
      li.textContent = 'Prompt fulfills enterprise quality standards with clear objectives.';
      suggestionsList.appendChild(li);
    } else {
      sugs.forEach((s) => {
        const li = document.createElement('li');
        li.textContent = s;
        suggestionsList.appendChild(li);
      });
    }

    // Improved Prompt
    improvedPromptText.textContent = data.improved_prompt || data.prompt;

    // Smooth scroll to result
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // Copy improved prompt to clipboard
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const text = improvedPromptText.textContent;
      if (!text) return;
      navigator.clipboard.writeText(text).then(() => {
        const oldText = copyBtn.textContent;
        copyBtn.textContent = '✓ Copied!';
        setTimeout(() => {
          copyBtn.textContent = oldText;
        }, 2000);
      });
    });
  }

  // Use improved prompt
  if (useImprovedBtn) {
    useImprovedBtn.addEventListener('click', () => {
      const text = improvedPromptText.textContent;
      if (!text) return;
      promptInput.value = text;
      updateCounter();
      form.dispatchEvent(new Event('submit'));
    });
  }
});
