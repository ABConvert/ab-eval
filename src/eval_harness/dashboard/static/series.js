// One chart palette for the whole app, read off the stylesheet rather than declared
// again in a template. compare.html carried its own array of eight hard-coded hex
// values — a third palette, invisible to the sheet — and assigned by array position,
// so a model's colour changed when you changed which runs you were comparing.
//
// Two things fix that. Colours come from --chart-1..5, the tokens round 1 wrote for
// exactly this. And a series is chosen by its NAME, not by where it happens to sit in
// the list, so a model keeps the same colour and the same shape on every page and in
// every selection.
//
// Every hue is paired with a point shape, because the hues do not survive greyscale.
// Measured on the served stylesheet: --chart-2 (#ff8690) and --chart-3 (#3fb4e8) come
// out at relative luminance 0.403 and 0.395 — a gap of 0.008, which is nothing — and
// --chart-3 against --chart-5 is 0.074. The shape is the encoding; the colour is the
// fast read for those who have it.
(function (global) {
  var SHAPES = ['circle', 'triangle', 'rect', 'rectRot', 'crossRot'];
  var DASHES = [[], [7, 3], [2, 3], [9, 3, 2, 3], [14, 4]];
  var COUNT = SHAPES.length;

  function token(n) {
    var v = getComputedStyle(document.documentElement).getPropertyValue('--chart-' + n);
    return (v || '').trim() || '#8f9ba8';
  }

  // `order` is every model in the data, sorted, handed in by the page. Sorting the
  // names on the page instead would put a model at a different index in every
  // selection — opus-5 second beside fable and first beside sonnet — so it would
  // change colour when you changed which runs you compared. The mark is only worth
  // learning if it is the same mark next time.
  function assign(names, order) {
    order = (order && order.length) ? order : names.slice().sort();
    var out = {};
    names.forEach(function (name) {
      var i = order.indexOf(name);
      if (i < 0) { i = COUNT - 1; }  // a model the page knows and the data root does not
      out[name] = {
        index: i + 1,            // 1-based, matching --chart-N and the .s-N classes
        color: token((i % COUNT) + 1),
        pointStyle: SHAPES[i % COUNT],
        dash: DASHES[i % COUNT],
        // Past five the shapes repeat at half weight rather than inventing a sixth hue.
        faded: i >= COUNT
      };
    });
    return out;
  }

  global.Series = {assign: assign, SHAPES: SHAPES, COUNT: COUNT};
})(window);
