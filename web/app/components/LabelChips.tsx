"use client";

export type Species = "cattle" | "buffalo";
export type LumpsAnswer = "yes" | "no" | "not_sure";

/**
 * The two questions this whole app exists to ask.
 *
 * The model's largest gap is diseased BUFFALO, which appear in no public
 * dataset anywhere (MODEL_CARD.md section 6). Every buffalo photo with a
 * human "yes, it has lumps" attached to it is worth more than any amount of
 * further engineering on the cattle data.
 *
 * So these are asked BEFORE the verdict is revealed. A person who has already
 * seen the model say "possible skin condition" is no longer an independent
 * observer -- they are agreeing or disagreeing with a machine, and the label
 * is contaminated.
 */
export function LabelChips({
  species,
  lumps,
  onSpecies,
  onLumps,
}: {
  species: Species | null;
  lumps: LumpsAnswer | null;
  onSpecies: (s: Species) => void;
  onLumps: (l: LumpsAnswer) => void;
}) {
  return (
    <div className="space-y-4">
      <Group
        label="Which animal is this?"
        labelUr="یہ کون سا جانور ہے؟"
        options={[
          { value: "cattle", label: "Cattle", labelUr: "گائے" },
          { value: "buffalo", label: "Buffalo", labelUr: "بھینس" },
        ]}
        value={species}
        onChange={(v) => onSpecies(v as Species)}
      />

      <Group
        label="Looking at the animal yourself — does it have lumps on its skin?"
        labelUr="آپ خود دیکھ کر بتائیں — کیا جانور کی جلد پر گلٹیاں ہیں؟"
        options={[
          { value: "yes", label: "Yes", labelUr: "ہاں" },
          { value: "no", label: "No", labelUr: "نہیں" },
          { value: "not_sure", label: "Not sure", labelUr: "پتہ نہیں" },
        ]}
        value={lumps}
        onChange={(v) => onLumps(v as LumpsAnswer)}
      />
    </div>
  );
}

function Group({
  label,
  labelUr,
  options,
  value,
  onChange,
}: {
  label: string;
  labelUr: string;
  options: { value: string; label: string; labelUr: string }[];
  value: string | null;
  onChange: (v: string) => void;
}) {
  return (
    <fieldset>
      <legend className="text-[15px] font-medium">{label}</legend>
      <p className="ur text-[14px]" style={{ color: "var(--muted)" }}>
        {labelUr}
      </p>
      <div className="mt-2 flex flex-wrap gap-2">
        {options.map((o) => {
          const active = value === o.value;
          return (
            <button
              key={o.value}
              type="button"
              onClick={() => onChange(o.value)}
              aria-pressed={active}
              className="tap rounded-full border px-5 text-[15px] font-medium transition-colors"
              style={{
                borderColor: active ? "var(--accent)" : "var(--line)",
                background: active ? "var(--accent)" : "transparent",
                color: active ? "var(--bg)" : "var(--fg)",
              }}
            >
              {o.label}
              <span className="ur ml-2 opacity-80">{o.labelUr}</span>
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}
