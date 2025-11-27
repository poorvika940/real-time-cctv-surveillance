// Service Worker to handle notification clicks and forward alert payloads to clients
self.addEventListener('notificationclick', function(event) {
  const n = event.notification;
  const action = event.action;
  // Try to focus an existing client or open a new one
  event.waitUntil((async () => {
    // If user clicked the 'Add Face' action, attempt an auto-add on the server using alert index
    if (action === 'add' && n && n.data && typeof n.data.idx !== 'undefined') {
      try {
        const idx = n.data.idx;
        const now = new Date();
        const pad = (x)=>x.toString().padStart(2,'0');
        const defName = `Person_${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
        await fetch('/add_face_from_alert', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ alert_idx: idx, name: defName, profession: 'Unknown' })
        }).catch(()=>{});
        // optional: show a quick confirmation notification
        try {
          await self.registration.showNotification('Face added', { body: `Saved as ${defName}`, requireInteraction: false });
        } catch(e) {}
      } catch (e) {
        // ignore
      }
    }
    const all = await clients.matchAll({ type: 'window', includeUncontrolled: true });
    let client = null;
    if (all && all.length) {
      // pick focused or first
      client = all.find(c => c.focused) || all[0];
      try { await client.focus(); } catch(e) {}
    }
    // If no client, open the stream page (camera 0) so we can forward the alert payload
    if (!client) {
      client = await clients.openWindow('/stream?camera=0');
    }
    // If client exists, post a message with the alert data
    try {
      if (client && n.data) {
        client.postMessage({ type: 'alert_notification', data: n.data, action: action || null });
      }
    } catch (e) {
      // ignore
    }
    n.close();
  })());
});

self.addEventListener('notificationclose', function(event) {
  // Could be used to record dismissal metrics
});

// Handle incoming push messages (Web Push)
self.addEventListener('push', function(event) {
  // Edge-only simplified push: single actionable "Add Face" notification for unknown faces
  const DEFAULT_ICON = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8AIAwMAsgG8bKzT/QAAAABJRU5ErkJggg==';
  let payload = {};
  try { if (event.data) payload = event.data.json(); } catch (e) { payload = { title: 'Unknown face', body: 'Add this face?' }; }
  const title = payload.title || 'Unknown face';
  const opts = {
    body: 'Add this face to your dataset',
    icon: payload.icon || DEFAULT_ICON,
    data: payload,
    tag: 'edge-unknown-face',
    requireInteraction: true,
    actions: [ { action: 'add', title: 'Add Face' } ]
  };
  event.waitUntil(self.registration.showNotification(title, opts));
});
