// Read the chart from the published table so there is a single source of values.
(() => {
  const controls = document.querySelector('.chart-controls');
  const chart = document.querySelector('.comparison-chart');
  const table = document.querySelector('#dmc-table');
  if (!controls || !chart || !table) return;
  const methods = Array.from(table.tHead.rows[0].cells).slice(2).map(cell => cell.textContent);
  function render(domain) {
    const rows = Array.from(table.tBodies[0].rows).filter(row => row.dataset.domain === domain);
    const grid = document.createElement('div');
    grid.className = 'chart-grid';
    rows.forEach(row => {
      const values = Array.from(row.cells).slice(2).map(cell => parseFloat(cell.textContent));
      const bestBaseline = Math.max(...values.slice(1));
      const baselineName = methods[values.indexOf(bestBaseline, 1)];
      const item = document.createElement('div');
      item.className = 'chart-item';
      const title = document.createElement('h4');
      title.textContent = row.cells[1].textContent;
      item.append(title);
      [values[0], bestBaseline].forEach((value, i) => {
        const barRow = document.createElement('div');
        barRow.className = 'bar-row';
        barRow.setAttribute('aria-label', `${i ? baselineName : 'CARDPol'}: ${value} mean return`);
        const track = document.createElement('div');
        track.className = 'bar-track';
        const bar = document.createElement('span');
        bar.className = i ? 'bar baseline' : 'bar';
        bar.style.width = `${value / 10}%`; // DMC returns use a common 0–1000 scale.
        track.append(bar);
        const label = document.createElement('span');
        label.textContent = value;
        barRow.append(track, label);
        item.append(barRow);
      });
      const note = document.createElement('p');
      note.textContent = `Strongest baseline: ${baselineName}`;
      item.append(note);
      grid.append(item);
    });
    const legend = document.createElement('p');
    legend.className = 'chart-legend';
    const name = domain === 'cheetah' ? 'Cheetah' : 'Walker';
    legend.append(`${name} · Mean return · Shared scale: 0–1000`);
    [['', 'CARDPol'], ['baseline', 'Strongest baseline']].forEach(([className, label]) => {
      const swatch = document.createElement('span');
      swatch.className = `legend-swatch ${className}`;
      swatch.setAttribute('aria-hidden', 'true');
      legend.append(swatch, label);
    });
    chart.replaceChildren(grid, legend);
    controls.querySelectorAll('button').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.domain === domain));
    });
  }
  controls.addEventListener('click', event => {
    const button = event.target.closest('button[data-domain]');
    if (button) render(button.dataset.domain);
  });
  render('cheetah');
  controls.hidden = false;
})();
