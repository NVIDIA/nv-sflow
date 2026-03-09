import React from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";

export default function Home() {
  return (
    <Layout title="sflow" description="Workflow Orchestrator with Pluggable Backends (local, Slurm, ...)">
      <main style={{ padding: "3rem 1.5rem", maxWidth: 960, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: "3rem" }}>
          <h1 style={{ fontSize: "3rem", marginBottom: "1rem", color: "#2e8555" }}>sflow</h1>
          <p style={{ fontSize: "1.3rem", marginBottom: "2rem", color: "#666" }}>
            A flexible workflow orchestrator with pluggable backends
          </p>
          <p style={{ fontSize: "1.1rem", lineHeight: "1.6", maxWidth: "600px", margin: "0 auto" }}>
            Define workflows declaratively in YAML, run them locally or on a cluster (e.g. Slurm), and track execution
            with built-in logging and TUI monitoring.
          </p>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: "2rem",
            marginBottom: "3rem",
          }}
        >
          <div style={{ padding: "1.5rem", border: "1px solid #e3e3e3", borderRadius: "8px" }}>
            <h3 style={{ color: "#2e8555", marginBottom: "1rem" }}>🚀 Easy to Use</h3>
            <p>Define workflows in simple YAML files with minimal configuration required.</p>
          </div>
          <div style={{ padding: "1.5rem", border: "1px solid #e3e3e3", borderRadius: "8px" }}>
            <h3 style={{ color: "#2e8555", marginBottom: "1rem" }}>⚡ Pluggable Backends</h3>
            <p>Switch between local and cluster execution by selecting a backend (e.g. Slurm) in config.</p>
          </div>
          <div style={{ padding: "1.5rem", border: "1px solid #e3e3e3", borderRadius: "8px" }}>
            <h3 style={{ color: "#2e8555", marginBottom: "1rem" }}>📊 Real-time Monitoring</h3>
            <p>Track your workflows with built-in TUI and comprehensive logging capabilities.</p>
          </div>
        </div>

        <div style={{ textAlign: "center" }}>
          <div style={{ display: "flex", gap: "1rem", justifyContent: "center", flexWrap: "wrap" }}>
            <Link className="button button--primary button--lg" to="/docs/user/quickstart">
              Get Started
            </Link>
            <Link className="button button--secondary button--lg" to="/docs/user/intro">
              Read the Docs
            </Link>
          </div>
        </div>

        <div style={{ marginTop: "3rem", padding: "2rem", backgroundColor: "#f8f9fa", borderRadius: "8px" }}>
          <h3 style={{ marginBottom: "1rem" }}>Quick Example</h3>
          <pre
            style={{
              backgroundColor: "#2d3748",
              color: "#e2e8f0",
              padding: "1rem",
              borderRadius: "4px",
              overflow: "auto",
            }}
          >
            {`version: "0.1"

workflow:
  name: hello_world
  tasks:
    - name: hello
      script:
        - echo "Hello from sflow!"`}
          </pre>
          <p style={{ marginTop: "1rem", fontSize: "0.9rem", color: "#666" }}>
            Run with: <code>sflow run --file workflow.yaml --tui</code>
          </p>
        </div>
      </main>
    </Layout>
  );
}
