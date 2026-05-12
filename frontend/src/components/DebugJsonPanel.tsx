import { useEffect, useMemo, useState } from "react";

type DebugJsonPanelProps = {
  value: unknown;
};

export function DebugJsonPanel({ value }: DebugJsonPanelProps) {
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "error">(
    "idle",
  );
  const jsonValue = useMemo(() => JSON.stringify(value, null, 2), [value]);

  useEffect(() => {
    if (copyStatus === "idle") {
      return;
    }

    const timeoutId = window.setTimeout(() => setCopyStatus("idle"), 1500);
    return () => window.clearTimeout(timeoutId);
  }, [copyStatus]);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(jsonValue);
      setCopyStatus("copied");
    } catch {
      setCopyStatus("error");
    }
  }

  return (
    <section className="panel">
      <div className="debug-json__header">
        <h2>Raw JSON State</h2>
        <div className="debug-json__actions">
          <button type="button" onClick={handleCopy}>
            Copy JSON
          </button>
          <span className="muted" aria-live="polite">
            {copyStatus === "copied"
              ? "Copied"
              : copyStatus === "error"
                ? "Copy failed"
                : ""}
          </span>
        </div>
      </div>
      <pre className="debug-json">{jsonValue}</pre>
    </section>
  );
}
