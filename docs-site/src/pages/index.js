import React from "react";
import Head from "@docusaurus/Head";
import useBaseUrl from "@docusaurus/useBaseUrl";
import useIsBrowser from "@docusaurus/useIsBrowser";
import { useLocation } from "@docusaurus/router";

// The homepage is a full-viewport frame around the static intro deck. Language is
// carried in the query string (`/?lang=zh`) rather than by swapping the iframe in
// place, so the Chinese deck is shareable, bookmarkable and survives a reload --
// an in-place swap leaves the address bar on "/" and silently reverts on refresh.
//
// The deck itself drives this: when it detects it is framed, its language toggle
// rewrites the TOP url instead of navigating its own document. See the
// `langToggle` handler in static/sflow_intro*.html.
const DECKS = {
  en: {
    file: "/sflow_intro.html",
    title: "NV-sflow — Declarative Workflow Descriptor",
    description:
      "Declarative workflow descriptor with swappable backends. Describe once, run anywhere.",
    frameTitle: "NV-sflow Introduction",
    htmlLang: "en",
  },
  zh: {
    file: "/sflow_intro_zh.html",
    title: "NV-sflow — 面向大规模 GPU 集群的声明式工作流描述器",
    description: "声明式工作流描述器，后端可自由切换。一次描述，随处运行。",
    frameTitle: "NV-sflow 介绍",
    htmlLang: "zh-CN",
  },
};

const DECK_BG = "#060a10"; // matches the deck's own background, so the pre-load frame is invisible

export default function Home() {
  const { search, hash } = useLocation();
  const isBrowser = useIsBrowser();

  const lang = new URLSearchParams(search).get("lang") === "zh" ? "zh" : "en";
  const deck = DECKS[lang];

  // Both are resolved unconditionally: useBaseUrl is a hook and cannot be called
  // behind a branch without breaking the rules of hooks.
  const enUrl = useBaseUrl(DECKS.en.file);
  const zhUrl = useBaseUrl(DECKS.zh.file);

  // Forward the slide anchor (#s7) so switching language deep in the deck lands
  // on the same slide instead of resetting to the title.
  const slide = /^#s[0-9a-z]+$/i.test(hash) ? hash : "";
  const src = (lang === "zh" ? zhUrl : enUrl) + slide;

  return (
    <>
      <Head>
        <html lang={deck.htmlLang} />
        <title>{deck.title}</title>
        <meta name="description" content={deck.description} />
        <link rel="alternate" hrefLang="en" href={enUrl} />
        <link rel="alternate" hrefLang="zh-Hans" href={zhUrl} />
        <style>{`
          .navbar, .footer, .main-wrapper > nav { display: none !important; }
          #__docusaurus { height: 100vh; overflow: hidden; }
          .main-wrapper { height: 100vh; padding: 0 !important; margin: 0 !important; }
        `}</style>
      </Head>
      <main
        style={{
          width: "100vw",
          height: "100vh",
          margin: 0,
          padding: 0,
          background: DECK_BG,
        }}
      >
        {/* Rendered only in the browser, and never during the hydration pass.
            The query string is not knowable at build time, so a server-rendered
            frame would always be the English deck -- and React does not patch an
            iframe `src` while hydrating, which stranded `?lang=zh` on the English
            deck. Waiting one tick costs nothing visible: `main` already paints the
            deck's own background colour. */}
        {isBrowser && (
          <iframe
            // Keyed by src so a language switch remounts the frame instead of
            // mutating an already-loaded one.
            key={src}
            src={src}
            title={deck.frameTitle}
            style={{
              width: "100%",
              height: "100%",
              border: "none",
              display: "block",
            }}
            allowFullScreen
          />
        )}
        {/* Without JS the frame above never renders, which would leave the
            homepage blank. `?lang=zh` is unreadable here, so this falls back to
            the English deck -- both decks link to each other anyway. */}
        <noscript>
          <iframe
            src={enUrl}
            title={DECKS.en.frameTitle}
            style={{
              width: "100%",
              height: "100%",
              border: "none",
              display: "block",
            }}
          />
        </noscript>
      </main>
    </>
  );
}
