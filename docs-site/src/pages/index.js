import React from "react";
import Head from "@docusaurus/Head";
import useBaseUrl from "@docusaurus/useBaseUrl";

export default function Home() {
  const introUrl = useBaseUrl("/sflow_intro.html");

  return (
    <>
      <Head>
        <title>NV-sflow — Declarative Workflow Descriptor</title>
        <meta
          name="description"
          content="Declarative workflow descriptor with swappable backends. Describe once, run anywhere."
        />
        <style>{`
          .navbar, .footer, .main-wrapper > nav { display: none !important; }
          #__docusaurus { height: 100vh; overflow: hidden; }
          .main-wrapper { height: 100vh; padding: 0 !important; margin: 0 !important; }
        `}</style>
      </Head>
      <main style={{ width: "100vw", height: "100vh", margin: 0, padding: 0 }}>
        <iframe
          src={introUrl}
          title="NV-sflow Introduction"
          style={{
            width: "100%",
            height: "100%",
            border: "none",
            display: "block",
          }}
          allowFullScreen
        />
      </main>
    </>
  );
}
