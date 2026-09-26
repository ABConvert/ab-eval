// Hide the model-form boxes the chosen provider cannot send, and follow the Provider select.
//
// The server already renders the right set for the provider on disk; this is what keeps it
// right while someone is changing the provider, before any save. A field that saves a value
// nothing sends is the defect the whole page is arguing against, so the boxes go away rather
// than being labelled and left clickable.
//
// Hidden inputs are also disabled, because a hidden box still posts. With JavaScript off
// nothing here runs, every box shows, and the handler drops what the provider cannot use —
// so the form is never more permissive than the config loader.
(function () {
  "use strict";

  // Mirrors PROVIDER_FIELDS in dashboard/config_io.py. Two copies of one fact, so the page
  // renders them into the document rather than either side guessing: see setup.html.
  var TABLE = {};
  try {
    var el = document.getElementById("provider-fields");
    TABLE = JSON.parse((el && el.textContent) || "{}");
  } catch (e) {
    return; // no table, no gating: leave every box visible rather than hide one wrongly
  }
  var ALL = TABLE.__all__ || [];

  function usable(provider) {
    return Object.prototype.hasOwnProperty.call(TABLE, provider) ? TABLE[provider] : ALL;
  }

  function apply(form) {
    var select = form.querySelector('select[name="provider"]');
    if (!select) return;
    var allowed = usable(select.value);
    form.querySelectorAll(".only-for").forEach(function (group) {
      var needs = (group.dataset.for || "").split(/\s+/).filter(Boolean);
      var show = needs.some(function (name) {
        return allowed.indexOf(name) !== -1;
      });
      group.hidden = !show;
      group.querySelectorAll("input, select, textarea").forEach(function (field) {
        field.disabled = !show;
      });
    });
  }

  function init() {
    document.querySelectorAll("form.model-form").forEach(apply);
  }

  document.addEventListener("change", function (ev) {
    var target = ev.target;
    if (target && target.name === "provider") {
      var form = target.closest("form.model-form");
      if (form) apply(form);
    }
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
