"use client";

export type SaveState = "idle" | "saving" | "saved" | "failed" | "unconfigured";

/**
 * Replaces the Send button. Saving is automatic now; this line is the honest
 * part of that -- it says what happened, and offers a retry when it did not.
 */
export function SaveStatus({
  photo,
  answer,
  error,
  onRetry,
}: {
  photo: SaveState;
  answer: SaveState;
  error?: string | null;
  onRetry?: () => void;
}) {
  if (photo === "unconfigured") {
    return (
      <p className="text-[12px]" style={{ color: "var(--muted)" }}>
        Saving is not configured for this build — nothing is being sent anywhere.
      </p>
    );
  }

  const failed = photo === "failed" || answer === "failed";

  let text: string;
  let textUr: string;
  if (photo === "saving") {
    text = "Saving the photo…";
    textUr = "تصویر محفوظ ہو رہی ہے…";
  } else if (failed) {
    text = "Could not save.";
    textUr = "محفوظ نہیں ہو سکی۔";
  } else if (answer === "saving") {
    text = "Photo saved · saving your answer…";
    textUr = "تصویر محفوظ · جواب محفوظ ہو رہا ہے…";
  } else if (answer === "saved") {
    text = "Photo and your answer saved — thank you.";
    textUr = "تصویر اور آپ کا جواب محفوظ — شکریہ۔";
  } else if (photo === "saved") {
    text = "Photo saved.";
    textUr = "تصویر محفوظ۔";
  } else {
    return null;
  }

  return (
    <div
      className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg px-3 py-2 text-[13px]"
      style={{
        background: failed ? "var(--concern-mid-bg)" : "var(--card)",
        color: failed ? "var(--concern-mid)" : "var(--muted)",
      }}
      role="status"
    >
      <span>{text}</span>
      <span className="ur">{textUr}</span>
      {failed && (
        <>
          {error && <span className="basis-full text-[12px] opacity-80">{error}</span>}
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="underline underline-offset-2"
            >
              Retry · دوبارہ
            </button>
          )}
        </>
      )}
    </div>
  );
}
