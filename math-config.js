window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]']],
    tags: 'ams',
    macros: {independent: '\\perp\\!\\!\\!\\perp'}
  },
  svg: {fontCache: 'local', scale: 1},
  options: {enableMenu: false, renderActions: {addMenu: [], assistiveMml: []}},
  startup: {
    ready() {
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(() => {
        document.querySelectorAll('.math').forEach(el => {
          const svg = el.querySelector('svg');
          if (svg) svg.setAttribute('aria-label', el.getAttribute('data-tex'));
        });
        document.querySelectorAll('mjx-container[display="true"]').forEach(el => {
          el.setAttribute('tabindex', '0');
          el.setAttribute('aria-label', 'Mathematical equation; scroll horizontally if needed');
        });
        const fitInlineMath = () => {
          document.querySelectorAll('.book-content .math.inline').forEach(el => {
            const svg = el.querySelector('svg');
            const available = el.closest('.book-content').clientWidth;
            const long = svg && svg.getBoundingClientRect().width > available - 4;
            el.classList.toggle('math-scroll', long);
            if (long) el.setAttribute('tabindex', '0');
            else el.removeAttribute('tabindex');
          });
        };
        fitInlineMath();
        window.addEventListener('resize', fitInlineMath);
        document.body.setAttribute('data-math-ready', 'true');
      });
    }
  }
};
