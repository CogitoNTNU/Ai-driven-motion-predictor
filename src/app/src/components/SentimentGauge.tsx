import { useState, useEffect, useRef } from "react";

interface ArticleEvent {
  headline: string;
  net_score: number;
  label: string;
  from_cache: boolean;
  current: number;
  total: number;
}

interface DoneEvent {
  type: "done";
  avg_score: number | null;
  label: string;
  article_count: number;
}

interface SentimentGaugeProps {
  ticker: string;
}

function interpolateColor(t: number): string {
  let r: number, g: number, b: number;
  if (t <= 0.5) {
    const s = t / 0.5;
    r = Math.round(239 + (234 - 239) * s);
    g = Math.round(68 + (179 - 68) * s);
    b = Math.round(68 + (8 - 68) * s);
  } else {
    const s = (t - 0.5) / 0.5;
    r = Math.round(234 + (34 - 234) * s);
    g = Math.round(179 + (197 - 179) * s);
    b = Math.round(8 + (94 - 8) * s);
  }
  return `rgb(${r},${g},${b})`;
}

const ARC_RADIUS = 80;
const CX = 100;
const CY = 100;
const ARC_LENGTH = Math.PI * ARC_RADIUS;

function SentimentSvg({ score }: { score: number }) {
  const t = (score + 1) / 2;
  const color = interpolateColor(t);
  const needleAngleDeg = t * 180;
  const needleAngleRad = (Math.PI * (180 - needleAngleDeg)) / 180;
  const needleLength = 70;
  const needleX = CX + needleLength * Math.cos(needleAngleRad);
  const needleY = CY - needleLength * Math.sin(needleAngleRad);
  const fillDash = t * ARC_LENGTH;
  const arcPath = `M ${CX - ARC_RADIUS} ${CY} A ${ARC_RADIUS} ${ARC_RADIUS} 0 0 1 ${CX + ARC_RADIUS} ${CY}`;

  return (
    <svg width="200" height="120" viewBox="0 0 200 120" className="mx-auto block">
      <path d={arcPath} fill="none" stroke="#404040" strokeWidth="16" strokeLinecap="round" />
      <path
        d={arcPath}
        fill="none"
        stroke={color}
        strokeWidth="16"
        strokeLinecap="round"
        strokeDasharray={`${fillDash} ${ARC_LENGTH}`}
        style={{ transition: "stroke-dasharray 0.4s ease, stroke 0.4s ease" }}
      />
      <line
        x1={CX} y1={CY} x2={needleX} y2={needleY}
        stroke="white" strokeWidth="2" strokeLinecap="round"
        style={{ transition: "x2 0.4s ease, y2 0.4s ease" }}
      />
      <circle cx={CX} cy={CY} r={4} fill="white" />
      <text x="5" y="118" fontSize="10" fill="#9ca3af">Bearish</text>
      <text x="148" y="118" fontSize="10" fill="#9ca3af">Bullish</text>
    </svg>
  );
}

function ScoreLabel({ score }: { score: number }) {
  const isPos = score >= 0;
  return (
    <span className={`font-medium text-sm ${isPos ? "text-green-400" : "text-red-400"}`}>
      {isPos ? "+" : ""}{score.toFixed(3)}
    </span>
  );
}

export function SentimentGauge({ ticker }: SentimentGaugeProps) {
  const [liveScore, setLiveScore] = useState<number | null>(null);
  const [liveLabel, setLiveLabel] = useState<string>("");
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);
  const [recentArticles, setRecentArticles] = useState<ArticleEvent[]>([]);
  const [done, setDone] = useState(false);
  const [articleCount, setArticleCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const scoreAccRef = useRef<number[]>([]);

  useEffect(() => {
    setLiveScore(null);
    setLiveLabel("");
    setProgress(null);
    setRecentArticles([]);
    setDone(false);
    setArticleCount(0);
    setError(null);
    scoreAccRef.current = [];

    const url = `${import.meta.env.VITE_API_URL ?? ""}/api/stock/${ticker}/sentiment/stream`;
    const es = new EventSource(url);

    es.onmessage = (event) => {
      const data = JSON.parse(event.data) as { type: string } & Partial<ArticleEvent & DoneEvent>;

      if (data.type === "article") {
        const art = data as ArticleEvent;
        scoreAccRef.current.push(art.net_score);
        const avg =
          scoreAccRef.current.reduce((a, b) => a + b, 0) / scoreAccRef.current.length;
        setLiveScore(avg);
        setLiveLabel(art.label);
        setProgress({ current: art.current, total: art.total });
        setRecentArticles((prev) => [art, ...prev].slice(0, 4));
      } else if (data.type === "done") {
        const d = data as DoneEvent;
        setLiveScore(d.avg_score);
        setLiveLabel(d.label);
        setArticleCount(d.article_count);
        setDone(true);
        es.close();
      }
    };

    es.onerror = () => {
      setError("Failed to load sentiment data");
      setDone(true);
      es.close();
    };

    return () => es.close();
  }, [ticker]);

  const isLoading = liveScore === null && !done && !error;

  return (
    <div className="rounded-xl border border-[#4d4d4f] bg-[#2f2f2f] p-4">
      <h2 className="mb-3 text-sm font-semibold text-white">News Sentiment</h2>

      {error ? (
        <p className="text-sm text-[#9ca3af]">{error}</p>
      ) : isLoading ? (
        <div className="space-y-3">
          <div className="animate-pulse mx-auto h-[110px] w-[200px] rounded-lg bg-[#404040]" />
          <div className="animate-pulse mx-auto h-4 w-24 rounded bg-[#404040]" />
          <div className="animate-pulse mx-auto h-3 w-16 rounded bg-[#404040]" />
        </div>
      ) : liveScore === null ? (
        <p className="py-4 text-center text-sm text-[#9ca3af]">Sentiment data unavailable</p>
      ) : (
        <div className="text-center">
          <SentimentSvg score={liveScore} />
          <div className="mt-2 space-y-1">
            <p className="font-semibold text-white">{liveLabel}</p>
            <p className="text-sm text-[#9ca3af]">
              {Math.round(((liveScore + 1) / 2) * 100)}%
            </p>
            {done ? (
              <p className="text-sm text-[#9ca3af]">📰 {articleCount} articles</p>
            ) : progress ? (
              <p className="text-xs text-[#6b7280]">
                Analysing {progress.current}/{progress.total} articles…
              </p>
            ) : null}
          </div>

          {/* Live article feed — shown while streaming, hidden when done */}
          {!done && recentArticles.length > 0 && (
            <div className="mt-3 space-y-1 text-left">
              {recentArticles.map((a, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 rounded-lg bg-[#212121] px-2.5 py-1.5"
                  style={{ opacity: 1 - i * 0.2 }}
                >
                  <ScoreLabel score={a.net_score} />
                  <span className="truncate text-xs text-[#9ca3af]">{a.headline}</span>
                  {a.from_cache && (
                    <span className="shrink-0 rounded bg-[#2f2f2f] px-1 text-[10px] text-[#6b7280]">
                      cached
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
