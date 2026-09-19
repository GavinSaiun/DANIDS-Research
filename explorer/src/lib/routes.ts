export const routes = [
  { id: "overview", label: "Overview", number: "01" },
  { id: "transfer", label: "Cross-domain transfer", number: "02" },
  { id: "health", label: "Model health", number: "03" },
  { id: "recoverability", label: "Recoverability", number: "04" },
  { id: "provenance", label: "Evidence & provenance", number: "05" },
] as const;
export type Page = (typeof routes)[number]["id"];
export function currentPage(): Page | "not-found" {
  const id = window.location.hash.slice(2) || "overview";
  return routes.some((r) => r.id === id) ? (id as Page) : "not-found";
}
export const number = (n: number) => n.toLocaleString("en-US");
