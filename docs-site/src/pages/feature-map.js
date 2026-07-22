import React from "react";
import Head from "@docusaurus/Head";
import Layout from "@theme/Layout";
import useBaseUrl from "@docusaurus/useBaseUrl";

export default function FeatureMap() {
  // NOTE: the embedded HTML is named `feature-map-embed.html`, not `feature-map.html`.
  // On clean-URL hosts (e.g. `docusaurus serve`) `/feature-map.html` strips to `/feature-map`,
  // which is THIS page's route — so embedding it would recurse into this page forever.
  const featureMapUrl = useBaseUrl("/feature-map-embed.html");

  return (
    <Layout noFooter>
      <Head>
        <title>Sflow Feature Map</title>
        <meta
          name="description"
          content="Interactive sflow feature navigation map showing what features exist and when to use them."
        />
      </Head>
      <main style={{ width: "100%", height: "calc(100vh - 60px)", margin: 0, padding: 0 }}>
        <iframe
          src={featureMapUrl}
          title="Sflow Feature Map"
          style={{
            width: "100%",
            height: "100%",
            border: "none",
            display: "block",
          }}
          allowFullScreen
        />
      </main>
    </Layout>
  );
}
