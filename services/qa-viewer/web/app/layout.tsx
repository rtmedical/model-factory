import type { Metadata, Viewport } from "next";
import { Fraunces } from "next/font/google";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";

import { Providers } from "@/app/providers";

import "./globals.css";

const fraunces = Fraunces({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-fraunces",
  axes: ["opsz", "SOFT"],
});

export const metadata: Metadata = {
  title: "RT Medical · Model QA",
  description:
    "Quality assurance and evaluation viewer for the model-factory segmentation models. Research use only — not a medical device.",
  robots: { index: false, follow: false },
  // Favicon auto-generated from `app/icon.svg` by Next.js.
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0e17" },
  ],
};

const themeBootstrap = `
(function(){try{
  var t = localStorage.getItem('rt-theme');
  if (t === 'dark') document.documentElement.setAttribute('data-theme','dark');
  else document.documentElement.setAttribute('data-theme','light');
}catch(_e){document.documentElement.setAttribute('data-theme','light');}})();
`;

// Sets data-layout-left / data-layout-right / data-layout-focus on the
// <html> root before first paint so the workspace grid renders with the
// right column template synchronously. The Zustand store reads the same
// localStorage key on module init, so SSR/CSR stay aligned.
const layoutBootstrap = `
(function(){try{
  var raw = localStorage.getItem('rt-qa-layout');
  var l = true, r = true, f = false;
  if (raw) {
    var p = JSON.parse(raw);
    if (typeof p.leftSidebarOpen === 'boolean') l = p.leftSidebarOpen;
    if (typeof p.rightSidebarOpen === 'boolean') r = p.rightSidebarOpen;
    if (typeof p.viewerFocus === 'boolean') f = p.viewerFocus;
  }
  var root = document.documentElement;
  root.setAttribute('data-layout-left', l ? 'open' : 'closed');
  root.setAttribute('data-layout-right', r ? 'open' : 'closed');
  root.setAttribute('data-layout-focus', f ? 'on' : 'off');
}catch(_e){/* defaults will apply */}})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const fontVars = `${fraunces.variable} ${GeistSans.variable} ${GeistMono.variable}`;
  return (
    <html lang="en" className={fontVars} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeBootstrap }} />
        <script dangerouslySetInnerHTML={{ __html: layoutBootstrap }} />
      </head>
      <body className="min-h-screen bg-[var(--color-rt-paper)] text-[var(--color-rt-ink)] antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
