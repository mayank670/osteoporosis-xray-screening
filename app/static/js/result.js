// Result page: animate the score bars and re-colour the Grad-CAM overlay.
(function () {
  "use strict";

  // Score bars: the width comes from data-value (0..1). It is set here rather than
  // with an inline style="" attribute, because the Content-Security-Policy forbids those.
  document.querySelectorAll(".bar-fill[data-value]").forEach((bar) => {
    const value = Math.max(0, Math.min(1, Number(bar.dataset.value) || 0));
    requestAnimationFrame(() => { bar.style.width = `${(value * 100).toFixed(1)}%`; });
  });

  // Heatmap controls: opacity slider and colour map.
  const img = document.getElementById("overlay-img");
  const alpha = document.getElementById("alpha");
  const colormap = document.getElementById("colormap");
  const alphaValue = document.getElementById("alpha-value");
  if (!img || !alpha || !colormap) return;

  let timer = null;
  function update() {
    alphaValue.textContent = Number(alpha.value).toFixed(2);
    clearTimeout(timer);  // debounce: one request after the slider stops moving
    timer = setTimeout(() => {
      const params = new URLSearchParams({ alpha: alpha.value, colormap: colormap.value });
      img.src = `${img.dataset.baseUrl}?${params}`;
    }, 150);
  }
  alpha.addEventListener("input", update);
  colormap.addEventListener("change", update);
})();
