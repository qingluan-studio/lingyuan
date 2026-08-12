(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  // --- Chart 1: Loss Curves ---
  var chart1 = echarts.init(document.getElementById('chart-loss'), null, { renderer: 'svg' });

  var epochs = [1,2,3,4,5,6,7,8,9,10,11,12];

  // v1 data (10 epochs, pad to 12)
  var v1Tang = [1.84, 0.98, 0.96, 1.10, 0.96, 1.05, 1.00, 1.05, 0.95, 0.89, null, null];
  var v1Lunyu = [2.72, 1.84, 1.80, 1.94, 1.80, 1.77, 1.76, 1.65, 1.68, 1.62, null, null];
  var v1Shake = [4.90, 4.06, 3.79, 3.61, 3.47, 3.26, 3.16, 3.13, 3.05, 2.98, null, null];

  // v1.5 data (12 epochs)
  var v15Tang = [3.70, 2.89, 2.84, 2.77, 2.77, 2.82, 2.69, 2.76, 2.81, 2.68, 2.65, 2.71];
  var v15Lunyu = [5.32, 4.58, 4.26, 4.05, 3.87, 3.74, 3.58, 3.51, 3.51, 3.50, 3.40, 3.32];
  var v15Shake = [5.37, 4.25, 3.81, 3.53, 3.43, 3.30, 3.16, 3.12, 3.01, 2.91, 2.93, 2.90];

  // v2 data (12 epochs)
  var v2Tang = [6.26, 5.99, 5.63, 5.23, 4.99, 4.78, 4.80, 4.85, 4.75, 4.78, 4.77, 4.77];
  var v2Lunyu = [6.25, 6.19, 6.08, 5.96, 5.89, 5.80, 5.70, 5.66, 5.64, 5.63, 5.60, 5.61];
  var v2Shake = [6.23, 6.08, 5.81, 5.64, 5.54, 5.39, 5.25, 5.18, 5.10, 5.12, 5.11, 5.14];

  chart1.setOption({
    title: { text: '', left: 'center' },
    tooltip: { trigger: 'axis', appendToBody: true },
    legend: {
      data: ['唐诗 v1', '唐诗 v1.5', '唐诗 v2', '论语 v1', '论语 v1.5', '论语 v2', '莎翁 v1', '莎翁 v1.5', '莎翁 v2'],
      top: 0,
      textStyle: { color: muted, fontSize: 11 },
      type: 'scroll'
    },
    grid: { left: '8%', right: '5%', bottom: '10%', top: '18%' },
    xAxis: {
      type: 'category',
      data: epochs,
      name: 'Epoch',
      nameTextStyle: { color: muted },
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted }
    },
    yAxis: {
      type: 'value',
      name: 'Loss',
      nameTextStyle: { color: muted },
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    animation: false,
    series: [
      { name: '唐诗 v1', type: 'line', data: v1Tang, smooth: false,
        lineStyle: { color: '#6b7280', width: 1.5, type: 'dashed' },
        itemStyle: { color: '#6b7280' }, symbolSize: 4 },
      { name: '唐诗 v1.5', type: 'line', data: v15Tang, smooth: false,
        lineStyle: { color: accent2, width: 1.5, type: 'dashed' },
        itemStyle: { color: accent2 }, symbolSize: 4 },
      { name: '唐诗 v2', type: 'line', data: v2Tang, smooth: true,
        lineStyle: { color: accent, width: 2.5 },
        itemStyle: { color: accent }, symbolSize: 5 },
      { name: '论语 v1', type: 'line', data: v1Lunyu, smooth: false,
        lineStyle: { color: '#9ca3af', width: 1, type: 'dashed' },
        itemStyle: { color: '#9ca3af' }, symbolSize: 3 },
      { name: '论语 v1.5', type: 'line', data: v15Lunyu, smooth: false,
        lineStyle: { color: '#c4b5fd', width: 1, type: 'dashed' },
        itemStyle: { color: '#c4b5fd' }, symbolSize: 3 },
      { name: '论语 v2', type: 'line', data: v2Lunyu, smooth: true,
        lineStyle: { color: '#5eead4', width: 2 },
        itemStyle: { color: '#5eead4' }, symbolSize: 4 },
      { name: '莎翁 v1', type: 'line', data: v1Shake, smooth: false,
        lineStyle: { color: '#4b5563', width: 1, type: 'dashed' },
        itemStyle: { color: '#4b5563' }, symbolSize: 3 },
      { name: '莎翁 v1.5', type: 'line', data: v15Shake, smooth: false,
        lineStyle: { color: '#8b5cf6', width: 1, type: 'dashed' },
        itemStyle: { color: '#8b5cf6' }, symbolSize: 3 },
      { name: '莎翁 v2', type: 'line', data: v2Shake, smooth: true,
        lineStyle: { color: '#14b8a6', width: 2 },
        itemStyle: { color: '#14b8a6' }, symbolSize: 4 }
    ]
  });
  window.addEventListener('resize', function() { chart1.resize(); });

  // --- Chart 2: UNK Ratio ---
  var chart2 = echarts.init(document.getElementById('chart-unk'), null, { renderer: 'svg' });
  chart2.setOption({
    tooltip: { trigger: 'axis', appendToBody: true, axisPointer: { type: 'shadow' } },
    legend: {
      data: ['v1', 'v1.5', 'v2'],
      top: 0,
      textStyle: { color: muted, fontSize: 12 }
    },
    grid: { left: '10%', right: '5%', bottom: '10%', top: '15%' },
    xAxis: {
      type: 'category',
      data: ['唐诗', '论语', '莎士比亚'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: ink, fontSize: 13 }
    },
    yAxis: {
      type: 'value',
      name: 'UNK 比例 (%)',
      nameTextStyle: { color: muted },
      max: 100,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, formatter: '{value}%' },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    animation: false,
    series: [
      {
        name: 'v1',
        type: 'bar',
        data: [98, 95, 50],
        itemStyle: { color: '#6b7280', borderRadius: [4, 4, 0, 0] },
        barWidth: '18%'
      },
      {
        name: 'v1.5',
        type: 'bar',
        data: [80, 60, 10],
        itemStyle: { color: accent2, borderRadius: [4, 4, 0, 0] },
        barWidth: '18%'
      },
      {
        name: 'v2',
        type: 'bar',
        data: [27.9, 6.5, 0],
        itemStyle: { color: accent, borderRadius: [4, 4, 0, 0] },
        barWidth: '18%',
        label: {
          show: true,
          position: 'top',
          color: accent,
          fontSize: 11,
          formatter: function(p) {
            return p.value > 0 ? p.value + '%' : '0%';
          }
        }
      }
    ]
  });
  window.addEventListener('resize', function() { chart2.resize(); });

  // --- Chart 3: Learning Rate Schedule ---
  var chart3 = echarts.init(document.getElementById('chart-lr'), null, { renderer: 'svg' });

  var lrSteps = [];
  var lrValues = [];
  var totalSteps = 300;
  var baseLr = 0.01;
  var warmupRatio = 0.1;
  var warmupSteps = Math.max(1, Math.floor(totalSteps * warmupRatio));

  for (var step = 0; step < totalSteps; step++) {
    lrSteps.push(step + 1);
    var lr;
    if (step < warmupSteps) {
      lr = baseLr * (step + 1) / warmupSteps;
    } else {
      var progress = (step - warmupSteps) / Math.max(1, totalSteps - warmupSteps);
      lr = baseLr * 0.5 * (1.0 + Math.cos(Math.PI * progress));
    }
    lrValues.push(parseFloat(lr.toFixed(6)));
  }

  chart3.setOption({
    tooltip: { trigger: 'axis', appendToBody: true, formatter: function(p) {
      return 'Step: ' + p[0].axisValue + '<br/>LR: ' + p[0].data;
    }},
    grid: { left: '10%', right: '5%', bottom: '10%', top: '10%' },
    xAxis: {
      type: 'category',
      data: lrSteps,
      name: 'Step',
      nameTextStyle: { color: muted },
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, interval: 29 }
    },
    yAxis: {
      type: 'value',
      name: 'Learning Rate',
      nameTextStyle: { color: muted },
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, formatter: function(v) { return v.toFixed(4); } },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    animation: false,
    series: [{
      type: 'line',
      data: lrValues,
      smooth: true,
      showSymbol: false,
      lineStyle: { color: accent, width: 2.5 },
      areaStyle: {
        color: {
          type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(45, 212, 191, 0.3)' },
            { offset: 1, color: 'rgba(45, 212, 191, 0.0)' }
          ]
        }
      },
      markLine: {
        silent: true,
        symbol: 'none',
        lineStyle: { color: accent2, type: 'dashed', width: 1 },
        data: [
          { xAxis: warmupSteps, label: { formatter: 'warmup end\n(step 30)', color: accent2, fontSize: 10, position: 'insideStartTop' } },
          { yAxis: baseLr, label: { formatter: 'peak LR=0.01', color: accent2, fontSize: 10, position: 'insideEndTop' } }
        ]
      }
    }]
  });
  window.addEventListener('resize', function() { chart3.resize(); });

})();
