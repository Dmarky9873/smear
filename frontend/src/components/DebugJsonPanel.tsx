type DebugJsonPanelProps = {
  value: unknown;
};

export function DebugJsonPanel({ value }: DebugJsonPanelProps) {
  return (
    <section className="panel">
      <h2>Raw JSON State</h2>
      <pre className="debug-json">{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}
