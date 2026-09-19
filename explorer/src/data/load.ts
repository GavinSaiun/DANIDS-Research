import {
  manifestSchema,
  schemas,
  type Evidence,
  type Manifest,
} from "../types/evidence";

export function canonical(text: string): Uint8Array<ArrayBuffer> {
  return new TextEncoder().encode(text.replace(/\r\n?/g, "\n"));
}
export async function sha256(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
}
export async function loadEvidence(
  get: (path: string) => Promise<string> = async (path) => {
    const response = await fetch(`${import.meta.env.BASE_URL}evidence/${path}`);
    if (!response.ok) throw new Error(`Evidence unavailable: ${path}`);
    return response.text();
  },
): Promise<{ evidence: Evidence; manifest: Manifest }> {
  const manifest = manifestSchema.parse(JSON.parse(await get("manifest.json")));
  const expected = Object.keys(schemas)
    .map((k) => `${k}.json`)
    .sort();
  if (
    JSON.stringify(manifest.files.map((f) => f.path).sort()) !==
    JSON.stringify(expected)
  ) {
    throw new Error("Unexpected evidence file roster");
  }
  const entries = await Promise.all(
    manifest.files.map(async (file) => {
      const text = await get(file.path);
      const bytes = canonical(text);
      if (
        bytes.length !== file.bytes ||
        (await sha256(bytes)) !== file.sha256
      ) {
        throw new Error(`Evidence integrity mismatch: ${file.path}`);
      }
      const key = file.path.replace(/\.json$/, "") as keyof Evidence;
      return [key, schemas[key].parse(JSON.parse(text))] as const;
    }),
  );
  return { evidence: Object.fromEntries(entries) as Evidence, manifest };
}
