// Presets are suggestions. The editable model field remains the only saved model value.
(function () {
  "use strict";
  function init(form) {
    var picker = form.querySelector(".model-preset");
    var provider = form.querySelector('[name="provider"]');
    var input = form.querySelector('[name="model"]');
    var key = form.querySelector('[name="key"]');
    if (!picker || !provider || !input) return;
    var presets;
    try { presets = JSON.parse(picker.dataset.presets); } catch (_) { return; }
    function sync() {
      picker.value = Array.from(picker.options).some(function (o) {
        return o.value === input.value;
      }) ? input.value : "";
    }
    provider.addEventListener("change", function () {
      picker.replaceChildren(new Option("Custom model / enter directly", ""));
      (presets[provider.value] || []).forEach(function (p) {
        picker.add(new Option(p[1], p[0]));
      });
      // Changing providers never silently replaces the user's identifier.
      sync();
    });
    picker.addEventListener("change", function () {
      if (picker.value) {
        input.value = picker.value;
        if (key && !key.value) key.value = picker.value;
      }
      input.focus();
    });
    input.addEventListener("input", sync);
    sync();
  }
  function initAll() { document.querySelectorAll("form.model-form").forEach(init); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initAll);
  else initAll();
})();
