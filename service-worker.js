self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open('trustnet-cache').then(function(cache) {
      return cache.addAll([
        '/',
        '/index.html',
        '/manifest.json'
      ]);
    })
  );
});