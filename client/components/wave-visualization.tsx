"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface WaveVisualizationProps {
  isActive: boolean;
  variant?: "user" | "ai";
  className?: string;
}

export function WaveVisualization({
  isActive,
  variant = "user",
  className,
}: WaveVisualizationProps) {
  const bars = Array.from({ length: 7 }, (_, i) => i);
  const [heights, setHeights] = useState<number[]>(bars.map(() => 4));

  useEffect(() => {
    if (!isActive) {
      setHeights(bars.map(() => 4));
      return;
    }

    const interval = setInterval(() => {
      setHeights(
        bars.map((_, i) => {
          const baseHeight = 20;
          const variation = Math.sin(Date.now() * 0.005 + i * 0.8) * 15;
          const random = (Math.random() - 0.5) * 10;
          return Math.max(8, baseHeight + variation + random);
        })
      );
    }, 50);

    return () => clearInterval(interval);
  }, [isActive, bars.length]);

  return (
    <div
      className={cn("flex items-end justify-center gap-1.5 h-20", className)}
    >
      {bars.map((bar, index) => (
        <div
          key={bar}
          className={cn(
            "w-2 rounded-full transition-all duration-150 ease-out",
            variant === "user"
              ? "bg-blue-500 dark:bg-blue-400"
              : "bg-purple-500 dark:bg-purple-400",
            !isActive && "opacity-30"
          )}
          style={{
            height: `${heights[index]}px`,
            minHeight: "4px",
          }}
        />
      ))}
    </div>
  );
}
