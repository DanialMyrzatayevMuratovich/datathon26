import { PageHeader, Section } from "@/components/kit";

// Placeholder: the demo is built during the datathon (skill ui-craft). Data comes from the FastAPI app via /api/*.
export default function Home() {
  return (
    <>
      <PageHeader title="Datathon'26" lead="Веб-демо решения появится здесь. API: uv run uvicorn app.main:app --port 8000" />
      <Section title="Источники">
        <ul className="list-disc pl-5 text-muted-foreground">
          <li>/api/runs — прогоны и метрики</li>
          <li>/api/runs/&#123;run_id&#125; — отчёт, шаги агента, графики</li>
          <li>/api/predict — предсказания лучшей модели</li>
        </ul>
      </Section>
    </>
  );
}
