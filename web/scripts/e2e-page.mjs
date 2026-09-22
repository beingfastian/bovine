/**
 * End-to-end walk through the real capture page in a real browser.
 *
 * /selftest proves the numbers are right. This proves the SCREEN is right:
 * that a picked file reaches the model, that the photo is saved the moment it
 * is scored, that species is pre-filled from the detector, that the label step
 * gates the reveal, that a verdict renders, that a non-animal is turned away
 * -- and, most importantly, that the forbidden words never appear on screen.
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
const stripAllowed = (text) => ALLOWED.reduce((acc, re) => acc.replace(re, ""), text);

const fixtures = JSON.parse(
  readFileSync(join("public", "fixtures", "fixtures.json"), "utf8")
).fixtures;

// One clear positive and one clear negative, away from every boundary.
const positive = [...fixtures].reverse().find((f) => f.expectedProbability > 0.95);
const negative = fixtures.find((f) => f.expectedProbability < 0.05);

// A real photograph of something that is not an animal. Windows ships
// landscape wallpapers; they are not committed, so this leg is skipped when
// the machine does not have them.
const NON_ANIMAL = [
  "C:\\Windows\\Web\\Wallpaper\\ThemeA\\img20.jpg",
  "C:\\Windows\\Web\\Wallpaper\\ThemeB\\img24.jpg",
  "C:\\Windows\\Web\\Wallpaper\\Spotlight\\img14.jpg",
].find(existsSync);

const failures = [];
function check(label, ok, detail = "") {
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${label}${detail ? " — " + detail : ""}`);
  if (!ok) failures.push(label);
}

const clickButton = (page, text) =>
  page.evaluate((t) => {
    const b = [...document.querySelectorAll("button")].find((x) =>
      x.textContent.trim().startsWith(t) || x.textContent.includes(t)
    );
    if (!b) return false;
    b.click();
    return true;
  }, text);

const bodyText = (page) => page.evaluate(() => document.body.innerText);

const waitForText = (page, re, timeout = 120_000) =>
  page.waitForFunction(
    (src, flags) => new RegExp(src, flags).test(document.body.innerText),
    { timeout, polling: 300 },
    re.source,
    re.flags
  );

const browser = await puppeteer.launch({
  executablePath,
  headless: "new",
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});

async function newPage() {
  const page = await browser.newPage();
  await page.setViewport({ width: 390, height: 844, isMobile: true });
  page.on("pageerror", (e) => console.error("  [page exception]", e.message));
  await page.goto(BASE + "/", { waitUntil: "networkidle2", timeout: 120_000 });
  const inputs = await page.$$('input[type="file"]');
  check("two file inputs present (camera + gallery)", inputs.length === 2);
  return { page, gallery: inputs[1] };
}

/** Whether this build talks to Supabase at all. */
async function saveConfigured(page) {
  const text = await bodyText(page);
  return !/not configured/i.test(text);
}

async function expectSaved(page, what) {
  if (!(await saveConfigured(page))) {
    console.log(`  [skip] ${what} — no Supabase credentials in this build`);
    return;
  }
  try {
    await waitForText(page, /Photo saved|Photo and your answer saved|Could not save/, 120_000);
  } catch {
    /* fall through to the check below */
  }
  const text = await bodyText(page);
  const saved = /Photo saved|Photo and your answer saved/.test(text);
  check(
    what,
    saved,
    saved
      ? ""
      : (text.match(/Could not save[\s\S]{0,240}/) ?? ["no save status rendered"])[0].replace(/\s+/g, " ")
  );
}

try {
  // ------------------------------------------------------------ cattle fixtures
  for (const fx of [positive, negative]) {
    console.log(`\n=== ${fx.file}  python p=${fx.expectedProbability} ===`);
    const { page, gallery } = await newPage();

    const before = await bodyText(page);
    check("disclosure about saving is visible before capture", /saved to improve the tool/i.test(before));
    check("no Send button exists", !/Send this photo/.test(before));

    await gallery.uploadFile(join("public", "fixtures", fx.file));
    await waitForText(page, /what do you see/, 180_000);
    check("label step gates the reveal", true);

    const labelling = await bodyText(page);
    check(
      "no verdict text leaks before labelling",
      !/Possible skin condition|No obvious skin lesion/.test(labelling)
    );

    // Species must be pre-filled by the detector for a cattle photo.
    // The chip's spans have no whitespace between them ("Cattleگائےdetected"),
    // so match on the prefix rather than splitting.
    const pressed = await page.evaluate(() =>
      [...document.querySelectorAll('button[aria-pressed="true"]')].map((b) => b.textContent.trim())
    );
    check(
      "species pre-filled from detector",
      pressed.some((t) => t.startsWith("Cattle") || t.startsWith("Buffalo")),
      pressed.join(" | ")
    );
    check("detector badge shown", /detected/i.test(labelling));

    // The photo save must not wait for any answer.
    await expectSaved(page, "photo saved before any answer");

    // Switch to buffalo to exercise the warning banner, then answer lumps.
    check("buffalo chip clickable", await clickButton(page, "Buffalo"));
    check("lumps chip clickable", await clickButton(page, "Yes"));
    check(
      "buffalo warning appears once buffalo is selected",
      /never been tested on a buffalo/i.test(await bodyText(page))
    );

    check("reveal button clickable", await clickButton(page, "Show the result"));
    await waitForText(
      page,
      /Possible skin condition|No obvious skin lesion in this photo|Unclear|too blurry|too dark|washed out|borderline/i,
      60_000
    );

    await clickButton(page, "Technical details");
    await new Promise((r) => setTimeout(r, 300));
    const text = await bodyText(page);

    const m = text.match(/raw score p\(lesion\)\s*\n?\s*([\d.]+)/);
    const shown = m ? Number(m[1]) : NaN;
    check(
      "raw score rendered and matches Python",
      Number.isFinite(shown) && Math.abs(shown - fx.expectedProbability) < 0.02,
      `browser ${shown} vs python ${fx.expectedProbability}`
    );
    check("species probabilities rendered", /p\(cattle\/buffalo\/other\)/.test(text));

    const expectPositive = fx.expectedProbability >= 0.6;
    check(
      "verdict matches the score",
      expectPositive
        ? /Possible skin condition/.test(text)
        : /No obvious skin lesion in this photo/.test(text)
    );

    const screened = stripAllowed(text);
    for (const re of FORBIDDEN) check(`forbidden copy absent: ${re}`, !re.test(screened));
    check("Urdu verdict rendered", /جلد|تصویر/.test(text));

    await expectSaved(page, "answer saved after reveal");
    await page.close();
  }

  // ------------------------------------------------------------ not an animal
  console.log(`\n=== non-animal: ${NON_ANIMAL ?? "(no wallpaper found)"} ===`);
  if (!NON_ANIMAL) {
    console.log("  [skip] no non-animal photo available on this machine");
  } else {
    const { page, gallery } = await newPage();
    await gallery.uploadFile(NON_ANIMAL);
    await waitForText(
      page,
      /could not find a cattle or buffalo|what do you see/i,
      180_000
    );
    const text = await bodyText(page);
    const rejected = /could not find a cattle or buffalo/i.test(text);
    check("non-animal photo is turned away by the gate", rejected);
    if (rejected) {
      check("no lesion verdict rendered for a non-animal", !/Possible skin condition|No obvious skin lesion/.test(text));
      check("asks whether an animal is actually present", /actually a cattle or buffalo/i.test(text));
      await expectSaved(page, "rejected photo still saved");
      check("'No' answer clickable", await clickButton(page, "No"));
      const screened = stripAllowed(await bodyText(page));
      for (const re of FORBIDDEN) check(`forbidden copy absent: ${re}`, !re.test(screened));
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
