/**
 * NFL Game Predictor — shared picks + leaderboard.
 *
 * Paste this into a Google Sheet (Extensions > Apps Script), then Deploy > New deployment > Web app,
 * "Execute as: Me", "Who has access: Anyone". Copy the web app URL into site_config.json in the repo.
 *
 * Picks are saved to the "Picks" tab of the sheet. Picks are only accepted BEFORE kickoff (checked here,
 * on Google's side, against the site's own schedule), and other people's picks are only shown once a
 * game has kicked off.
 */
const SCHEDULE_URL = 'https://zacharyivezi08.github.io/nfl-game-predictor/data/schedule.json';

function schedule_() {
  const cache = CacheService.getScriptCache();
  let s = cache.get('schedule');
  if (!s) {
    s = UrlFetchApp.fetch(SCHEDULE_URL, { muteHttpExceptions: true }).getContentText();
    cache.put('schedule', s, 600); // refresh every 10 minutes
  }
  return JSON.parse(s);
}

function sheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName('Picks');
  if (!sh) {
    sh = ss.insertSheet('Picks');
    sh.appendRow(['Saved at', 'Name', 'Game', 'Pick', 'Kickoff']);
  }
  return sh;
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function cleanName_(n) {
  n = String(n || '').replace(/\s+/g, ' ').trim().slice(0, 24);
  return /^[=+\-@]/.test(n) ? "'" + n : n; // stop spreadsheet formulas
}

function doPost(e) {
  try {
    const d = JSON.parse(e.postData.contents);
    const name = cleanName_(d.name);
    const game = String(d.game || '');
    const team = d.team ? String(d.team) : '';
    if (!name) return json_({ ok: false, error: 'Enter a name first' });
    const g = schedule_()[game];
    if (!g) return json_({ ok: false, error: 'Unknown game' });
    if (Date.now() >= Date.parse(g.kick)) return json_({ ok: false, error: 'Locked: the game has kicked off' });
    if (team && team !== g.home && team !== g.away) return json_({ ok: false, error: 'Not a team in this game' });

    const lock = LockService.getScriptLock();
    lock.waitLock(10000);
    try {
      const sh = sheet_();
      const vals = sh.getDataRange().getValues();
      const key = name.toLowerCase();
      for (let i = vals.length - 1; i >= 1; i--) { // one pick per person per game: replace the old one
        if (String(vals[i][1]).toLowerCase() === key && vals[i][2] === game) sh.deleteRow(i + 1);
      }
      if (team) sh.appendRow([new Date(), name, game, team, g.kick]);
    } finally {
      lock.releaseLock();
    }
    return json_({ ok: true });
  } catch (err) {
    return json_({ ok: false, error: String(err) });
  }
}

function doGet() {
  const sched = schedule_();
  const vals = sheet_().getDataRange().getValues();
  const now = Date.now();
  const picks = [];
  for (let i = 1; i < vals.length; i++) {
    const name = String(vals[i][1]).replace(/^'/, ''), game = vals[i][2], team = vals[i][3];
    const g = sched[game];
    if (g && now >= Date.parse(g.kick)) picks.push({ n: name, g: game, t: team }); // only locked games
  }
  return json_({ picks: picks });
}
