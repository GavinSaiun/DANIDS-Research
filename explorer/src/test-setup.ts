import { webcrypto } from "node:crypto";
import { TextEncoder, TextDecoder } from "node:util";
import { vi, afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

Object.defineProperty(globalThis, "crypto", {
  value: webcrypto,
  configurable: true,
});
Object.assign(globalThis, { TextEncoder, TextDecoder });
// jsdom has no layout. Supply a deterministic chart viewport rather than suppress warnings.
HTMLElement.prototype.getBoundingClientRect = () => new DOMRect(0, 0, 800, 310);
globalThis.ResizeObserver = class {
  constructor(private callback: ResizeObserverCallback) {}
  observe(target: Element) {
    this.callback(
      [
        {
          target,
          contentRect: target.getBoundingClientRect(),
        } as ResizeObserverEntry,
      ],
      this,
    );
  }
  unobserve() {}
  disconnect() {}
};
window.scrollTo = vi.fn();
afterEach(() => cleanup());
