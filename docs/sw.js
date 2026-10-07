// Lets the site install as an app. Always fetches fresh pages (picks update every 30 minutes);
// falls back to the last copy only when offline.
const C='nflgp-v1';
self.addEventListener('install',e=>self.skipWaiting());
self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));
self.addEventListener('fetch',e=>{
  if(e.request.method!=='GET'||new URL(e.request.url).origin!==location.origin)return;
  e.respondWith(fetch(e.request).then(r=>{const c=r.clone();caches.open(C).then(x=>x.put(e.request,c));return r;})
    .catch(()=>caches.match(e.request)));
});
