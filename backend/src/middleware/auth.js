/*
 Dev-bypass auth module — for local testing only.
 Exports named functions `requireAuth` and `requireRole` which simply attach
 a fake user to the request and continue.
 Replace with real auth for production and restore original file afterward.
*/

export function requireAuth(req, res, next) {
  // attach a fake user so routes that expect req.user work
  req.user = { id: 'dev', email: 'dev@local', role: 'admin' };
  return next();
}

export function requireRole(role) {
  // returns middleware that ensures req.user exists and optionally checks role
  return function (req, res, next) {
    req.user = req.user || { id: 'dev', email: 'dev@local', role: 'admin' };
    // if a specific role is required, bypass check in dev
    return next();
  }
}

// also export a default that some imports might expect (safe no-op)
export default requireAuth;
