// Relative timestamps, vendored: no library, no network.
//
// The server renders the exact UTC moment, so a page with no script still says
// when something happened. This rewrites each one as "6 days ago" and moves the
// exact moment into the title, because on these pages the useful question is
// almost always how long ago rather than precisely when.
(function () {
  var MINUTE = 60, HOUR = 3600, DAY = 86400;

  function phrase(seconds) {
    if (seconds < 45) { return 'just now'; }
    if (seconds < 90) { return 'a minute ago'; }
    if (seconds < HOUR) { return Math.round(seconds / MINUTE) + ' min ago'; }
    if (seconds < 2 * HOUR) { return 'an hour ago'; }
    if (seconds < DAY) {
      var h = Math.floor(seconds / HOUR);
      var m = Math.round((seconds - h * HOUR) / MINUTE);
      return m ? h + 'h ' + m + 'm ago' : h + 'h ago';
    }
    if (seconds < 2 * DAY) { return 'yesterday'; }
    return Math.round(seconds / DAY) + ' days ago';
  }

  function paint() {
    var now = Date.now();
    document.querySelectorAll('time[data-ts]').forEach(function (el) {
      var then = Date.parse(el.dataset.ts);
      if (isNaN(then)) { return; }
      if (!el.title) { el.title = el.textContent.trim(); }
      el.setAttribute('datetime', el.dataset.ts);
      el.textContent = phrase(Math.max(0, (now - then) / 1000));
    });
  }

  paint();
  // A run can be open for hours; "just now" should not still say that at teatime.
  setInterval(paint, 30000);
})();
