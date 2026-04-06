type DebugJsonPanelProps = {
  title?: string;
  value: unknown;
};

export function DebugJsonPanel({ title = "Raw JSON Debug", value }: DebugJsonPanelProps) {
  return (
    <section className="panel">
      <header className="panel-header">
        <div>
          <h2>{title}</h2>
          <p>Pretty-printed backend payload for debugging.</p>
        </div>
      </header>

      <pre className="json-panel">{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}
