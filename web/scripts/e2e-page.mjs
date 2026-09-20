/**
 * End-to-end walk through the real capture page in a real browser.
 *
 * /selftest proves the numbers are right. This proves the SCREEN is right:
 * that a picked file reaches the model, that the label step gates the reveal,
 * that a verdict renders -- and, most importantly, that the forbidden words
 * never appear on screen.
 *
 *   node scripts/e2e-page.mjs [baseUrl]
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
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
  console.error("No Chrome or Edge found.");
  process.exit(2);
}

/**
 * Words that must never reach a user's screen (MODEL_CARD.md section 2).
 *
 * The safety rules also REQUIRE the app to say it is "not a diagnosis", so
 * that exact phrase is whitelisted and stripped before the check runs.
 * Nothing else is. Any other appearance of these words is a regression, and
 * sanctioning one has to be a deliberate edit to this list rather than an
 * accident in a component.
 */
const ALLOWED = [/\bnot a diagnosis\b/gi];
const FORBIDDEN = [/\bhealthy\b/i, /lumpy\s*skin/i, /\bLSD\b/, /\bdiagnos(is|ed|e)\b/i];

function stripAllowed(text) {
  return ALLOWED.reduce((acc, re) => acc.replace(re, ""), text);
}

const fixtures = JSON.parse(
  readFileSync(join("public", "fixtures", "fixtures.json"), "utf8")
).fixtures;

// One clear positive and one clear negative, away from every boundary.
const positive = [...fixtures].reverse().find((f) => f.expectedProbability > 0.95);
const negative = fixtures.find((f) => f.expectedProbability < 0.05);

const failures = [];
function check(label, ok, detail = "") {
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${label}${detail ? " — " + detail : ""}`);
  if (!ok) failures.push(label);
}

const browser = await puppeteer.launch({
  executablePath,
  headless: "new",
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});

try {
  for (const fx of [positive, negative]) {
    console.log(`\n=== ${fx.file}  python p=${fx.expectedProbability} ===`);

    const page = await browser.newPage();
    await page.setViewport({ width: 390, height: 844, isMobile: true });
    page.on("pageerror", (e) => console.error("  [page exception]", e.message));

    await page.goto(BASE + "/", { waitUntil: "networkidle2", timeout: 120_000 });

    // The gallery input is the second hidden file input.
    const inputs = await page.$$('input[type="file"]');
    check("two file inputs present (camera + gallery)", inputs.length === 2);
    await inputs[1].uploadFile(join("public", "fixtures", fx.file));

    await page.waitForFunction(
      () => document.body.innerText.includes("what do you see"),
      { timeout: 180_000, polling: 300 }
    );
    check("label step gates the reveal", true);

    const beforeReveal = await page.evaluate(() => document.body.innerText);
    check(
      "no verdict text leaks before labelling",
      !/Possible skin condition|No obvious skin lesion/.test(beforeReveal)
    );

    // Answer both chips.
    await page.evaluate(() => {
      const byText = (t) =>
        [...document.querySelectorAll("button")].find((b) =>
          b.textContent.trim().startsWith(t)
        );
      byText("Buffalo")?.click();
    });
    await page.evaluate(() => {
      const byText = (t) =>
        [...document.querySelectorAll("button")].find((b) =>
          b.textContent.trim().startsWith(t)
        );
      byText("Yes")?.click();
    });

    const banner = await page.evaluate(() => document.body.innerText);
    check(
      "buffalo warning appears once buffalo is selected",
      /never been tested on a buffalo/i.test(banner)
    );

    await page.evaluate(() => {
      const b = [...document.querySelectorAll("button")].find((x) =>
        x.textContent.includes("Show the result")
      );
      b?.click();
    });

    await page.waitForFunction(
      () =>
        /Possible skin condition|No obvious skin lesion in this photo|Unclear|too blurry|too dark|washed out|borderline/i.test(
          document.body.innerText
        ),
      { timeout: 60_000, polling: 300 }
    );

    // Open the technical panel to read the raw score.
    await page.evaluate(() => {
      const b = [...document.querySelectorAll("button")].find((x) =>
        x.textContent.includes("Technical details")
      );
      b?.click();
    });
    await new Promise((r) => setTimeout(r, 300));

    const text = await page.evaluate(() => document.body.innerText);

    const m = text.match(/raw score p\(lesion\)\s*\n?\s*([\d.]+)/);
    const shown = m ? Number(m[1]) : NaN;
    check(
      "raw score rendered and matches Python",
      Number.isFinite(shown) && Math.abs(shown - fx.expectedProbability) < 0.02,
      `browser ${shown} vs python ${fx.expectedProbability}`
    );

    const expectPositive = fx.expectedProbability >= 0.6;
    check(
      "verdict matches the score",
      expectPositive
        ? /Possible skin condition/.test(text)
        : /No obvious skin lesion in this photo/.test(text)
    );

    const screened = stripAllowed(text);
    for (const re of FORBIDDEN) {
      check(`forbidden copy absent: ${re}`, !re.test(screened));
    }

    check("Urdu verdict rendered", /جلد|تصویر/.test(text));

    // --- submission, when the build is configured for it ---
    if (/Send this photo/.test(text)) {
      await page.evaluate(() => {
        const b = [...document.querySelectorAll("button")].find((x) =>
          x.textContent.includes("Send this photo")
        );
        b?.click();
      });
      await page.waitForFunction(
        () => /Sent — thank you|Photo upload failed|Saving the record failed|not configured/.test(
          document.body.innerText
        ),
        { timeout: 120_000, polling: 300 }
      );
      const after = await page.evaluate(() => document.body.innerText);
      const sent = /Sent — thank you/.test(after);
      check(
        "submission accepted by Supabase",
        sent,
        sent ? "" : (after.match(/(Photo upload failed|Saving the record failed)[^\n]*/) ?? ["see page"])[0]
      );
    } else if (/not configured/.test(text)) {
      console.log("  [skip] submission — no Supabase credentials in this build");
    }

    await page.close();
  }
} finally {
  await browser.close();
}

console.log("\n" + "=".repeat(60));
if (failures.length) {
  console.log("FAILED:", failures.length);
  for (const f of failures) console.log("  -", f);
  process.exitCode = 1;
} else {
  console.log("ALL CHECKS PASSED");
}
