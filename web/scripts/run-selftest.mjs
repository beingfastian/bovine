/**
 * Drive /selftest in a real Chrome and print the parity result.
 *
 * Uses puppeteer-core against the browser already installed on this machine --
 * no 130 MB Chromium download, which matters on this connection.
 *
 *   node scripts/run-selftest.mjs [baseUrl]
 *
 * Exits non-zero if the browser pipeline disagrees with Python.
 */
import { existsSync } from "node:fs";
import puppeteer from "puppeteer-core";

const BASE = process.argv[2] ?? "http://127.0.0.1:4173";

const CANDIDATES = [
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
];

const executablePath = CANDIDATES.find((p) => existsSync(p));
if (!executablePath) {
  console.error("No Chrome or Edge found. Open " + BASE + "/selftest by hand.");
  process.exit(2);
}
console.log("browser:", executablePath);

const browser = await puppeteer.launch({
  executablePath,
  headless: "new",
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});

try {
  const page = await browser.newPage();
  page.on("console", (m) => {
    if (m.type() === "error") console.error("  [page error]", m.text());
  });
  page.on("pageerror", (e) => console.error("  [page exception]", e.message));

  console.log("opening", BASE + "/selftest");
  await page.goto(BASE + "/selftest", { waitUntil: "networkidle2", timeout: 120_000 });

  // The page scores fixtures one at a time; wait for the verdict banner.
  await page.waitForFunction(
    () => /PASS|FAIL/.test(document.body.innerText),
    { timeout: 300_000, polling: 500 }
  );

  const result = await page.evaluate(() => {
    // Cells by header name, so adding a column to the page cannot silently
    // shift what this reads (it did: `ms` came back null after species/d²).
    const heads = [...document.querySelectorAll("thead th")].map((th) => th.textContent.trim());
    const col = (name) => heads.indexOf(name);
    const rows = [...document.querySelectorAll("tbody tr")].map((tr) => {
      const c = [...tr.children].map((td) => td.textContent.trim());
      return {
        file: c[col("file")], python: +c[col("python")], browser: +c[col("browser")],
        diff: +c[col("diff")], species: c[col("species")], pOther: +c[col("p_other")],
        d2: +c[col("d²")], ms: +c[col("ms")], verdict: c[col("verdict")],
      };
    });
    const banner = document.querySelector('[style*="border-radius"]');
    return { rows, text: document.body.innerText.split("\n").filter(Boolean).slice(0, 12) };
  });

  console.log("\n" + "=".repeat(74));
  console.log("BROWSER vs PYTHON");
  console.log("=".repeat(74));
  console.log(
    "  " + "file".padEnd(10) + "python".padStart(9) + "browser".padStart(10) + "diff".padStart(10) +
      "  species".padEnd(11) + "p_other".padStart(8) + "d²".padStart(7) + "ms".padStart(6) + "  verdict"
  );
  for (const r of result.rows) {
    console.log(
      "  " + r.file.padEnd(10) + r.python.toFixed(4).padStart(9) + r.browser.toFixed(4).padStart(10) +
        r.diff.toFixed(5).padStart(10) + ("  " + r.species).padEnd(11) + r.pOther.toFixed(3).padStart(8) +
        r.d2.toFixed(0).padStart(7) + String(r.ms).padStart(6) + "  " + r.verdict
    );
  }
  const speciesWrong = result.rows.filter((r) => r.species.includes("(py:")).length;
  console.log("  species disagreements:", speciesWrong);

  const diffs = result.rows.map((r) => r.diff);
  const maxDiff = Math.max(...diffs);
  const meanDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  const flips = result.rows.filter(
    (r) => r.verdict.includes("→") && !r.verdict.includes("(boundary)")
  ).length;
  const boundaryFlips = result.rows.filter((r) => r.verdict.includes("(boundary)")).length;
  const times = result.rows.map((r) => r.ms).sort((a, b) => a - b);

  console.log("\n  fixtures        ", result.rows.length);
  console.log("  max abs diff    ", maxDiff.toFixed(5));
  console.log("  mean abs diff   ", meanDiff.toFixed(5));
  console.log("  verdict flips   ", flips, `(+${boundaryFlips} on a threshold boundary, expected)`);
  console.log("  median latency  ", times[Math.floor(times.length / 2)], "ms");
  console.log("  max latency     ", times[times.length - 1], "ms");

  const pass = flips === 0 && maxDiff < 0.02 && speciesWrong === 0;
  console.log("\n  OVERALL:", pass ? "PASS" : "FAIL");
  if (!pass) process.exitCode = 1;
} finally {
  await browser.close();
}
