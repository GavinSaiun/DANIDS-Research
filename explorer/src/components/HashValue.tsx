import { useState } from "react";

export function HashValue({ value, label }: { value: string; label: string }) {
  const [message, setMessage] = useState("");
  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setMessage("Copied.");
    } catch {
      setMessage("Copy unavailable. Select and copy the hash above.");
    }
  }
  return (
    <div className="hash-value">
      <code>{value}</code>
      <button type="button" onClick={copy} aria-label={`Copy ${label}`}>
        Copy
      </button>
      <span className="copy-status" role="status">{message}</span>
    </div>
  );
}
