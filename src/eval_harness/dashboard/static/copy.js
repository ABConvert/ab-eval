// Clipboard, vendored: no library, no network. Shared by /setup and /, both of
// which hand the reader commands to paste into a terminal.
(function () {
  // Four templates load this file and only two carry #copy-status, so both branches of
  // the handler below were dereferencing null on /runs and /jobs/{id} — the only
  // uncaught exception in the product. A failed copy was silent there, and a successful
  // one would have thrown before the timeout that restores the button label, leaving it
  // reading "Copied" for the life of the page.
  function say(msg) {
    var status = document.getElementById('copy-status');
    if (status) { status.textContent = msg; }
  }

  // The buttons are inert without this file, so the markup ships them hidden and
  // they appear only once something can act on a click.
  document.querySelectorAll('.copy[hidden]').forEach(function (b) { b.hidden = false; });

  function write(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (err) { ok = false; }
      document.body.removeChild(ta);
      if (ok) { resolve(); } else { reject(new Error('copy refused')); }
    });
  }

  document.addEventListener('click', function (e) {
    var btn = e.target.closest ? e.target.closest('.copy') : null;
    if (!btn) { return; }
    var from = btn.getAttribute('data-copy-from');
    var source = from ? document.getElementById(from) : null;
    var text = from ? (source ? source.textContent : '') : btn.getAttribute('data-copy');
    if (!text) { return; }
    var label = btn.textContent;
    write(text).then(function () {
      btn.textContent = 'Copied';
      btn.classList.add('done');
      say('Copied to the clipboard');
      setTimeout(function () {
        btn.textContent = label;
        btn.classList.remove('done');
      }, 1600);
    }, function () {
      say('Could not copy — select the text and copy it by hand');
    });
  });
})();
