(() => {
  // These identifiers are completed once the GitHub repository and its
  // Discussions category have been created.  Keeping them here makes the
  // public comment system independent of the book's static files.
  const config = {
    repo: 'lmaowisc/missing_data',
    repoId: '',
    category: 'Comments',
    categoryId: '',
  };
  const container = document.getElementById('giscus-comments');
  if (!container) return;
  if (!config.repoId || !config.categoryId) {
    container.innerHTML = '<p>Comments will be available here once the public discussion forum is activated.</p>';
    return;
  }
  const script = document.createElement('script');
  script.src = 'https://giscus.app/client.js';
  script.async = true;
  script.crossOrigin = 'anonymous';
  script.setAttribute('data-repo', config.repo);
  script.setAttribute('data-repo-id', config.repoId);
  script.setAttribute('data-category', config.category);
  script.setAttribute('data-category-id', config.categoryId);
  script.setAttribute('data-mapping', 'pathname');
  script.setAttribute('data-strict', '1');
  script.setAttribute('data-reactions-enabled', '1');
  script.setAttribute('data-emit-metadata', '0');
  script.setAttribute('data-input-position', 'top');
  script.setAttribute('data-theme', 'light');
  script.setAttribute('data-lang', 'en');
  container.append(script);
})();
