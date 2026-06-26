import React from "react";
import Head from "@docusaurus/Head";
import Layout from "@theme/Layout";
import useBaseUrl from "@docusaurus/useBaseUrl";

export default function FeatureMap() {
  const featureMapUrl = useBaseUrl("/feature-map.html");

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
