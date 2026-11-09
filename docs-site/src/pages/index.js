import React from "react";
import Head from "@docusaurus/Head";
import useBaseUrl from "@docusaurus/useBaseUrl";
import useIsBrowser from "@docusaurus/useIsBrowser";
import { Redirect, useLocation } from "@docusaurus/router";

// The homepage is a full-viewport frame around the static feature video
// (static/sflow_landing.html, generated from the v0.3.2 feature-video page in
// docs/developer/dev-notes/v0.3.2-feature-video). Its links carry <base target="_top">,
// so docs, release notes and /intro open in the top window, not inside the frame.
//
// The intro deck that used to live here is now /intro. Links shared before the move
// (`/?lang=zh` for the Chinese deck, `/#s7` for a slide) are forwarded there.
const PAGE_BG = "#0b0a13"; // the page's own background, so the pre-load frame is invisible

export default function Home() {
  const { search, hash } = useLocation();
  const isBrowser = useIsBrowser();
  const pageUrl = useBaseUrl("/sflow_landing.html");
  const introUrl = useBaseUrl("/intro");

  const oldDeckLink = new URLSearchParams(search).has("lang") || /^#s[0-9a-z]+$/i.test(hash);
  if (isBrowser && oldDeckLink) {
    return <Redirect to={introUrl + search + hash} />;
  }

  const frame = {
    width: "100%",
    height: "100%",
    border: "none",
    display: "block",
  };

  return (
    <>
      <Head>
        <title>NV-sflow — Describe once. Run anywhere</title>
        <meta
          name="description"
          content="Write a GPU workload once and run it on a workstation, Docker hosts, Slurm or Kubernetes by swapping one backend file."
        />
        <style>{`
          .navbar, .footer, .main-wrapper > nav { display: none !important; }
          #__docusaurus { height: 100vh; overflow: hidden; }
          .main-wrapper { height: 100vh; padding: 0 !important; margin: 0 !important; }
        `}</style>
      </Head>
      <main style={{ width: "100vw", height: "100vh", margin: 0, padding: 0, background: PAGE_BG }}>
        {/* Browser-only, like /intro: the old-link forward above needs the real URL. */}
        {isBrowser && (
          // clipboard-write: the page's copy buttons (install command, agent prompt) run inside this frame
          <iframe src={pageUrl} title="sflow v0.3.2 feature video" style={frame} allow="clipboard-write" allowFullScreen />
        )}
        <noscript>
          <iframe src={pageUrl} title="sflow v0.3.2 feature video" style={frame} />
        </noscript>
      </main>
    </>
  );
}
