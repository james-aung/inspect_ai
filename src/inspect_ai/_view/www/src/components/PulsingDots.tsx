import clsx from "clsx";
import { FC } from "react";
import styles from "./PulsingDots.module.css";

interface PulsingDotsProps {
  text?: string;
  dotsCount?: number;
  subtle?: boolean;
  size?: "small" | "medium" | "large";
  className?: string | string[];
}

export const PulsingDots: FC<PulsingDotsProps> = ({
  text = "Loading...",
  dotsCount = 3,
  subtle = true,
  size = "small",
  className,
}) => {
  return (
    <div
      className={clsx(
        styles.container,
        size === "small"
          ? styles.small
          : size === "medium"
            ? styles.medium
            : styles.large,
        className,
      )}
      role="status"
    >
      <div className={styles.dotsContainer}>
        {[...Array(dotsCount)].map((_, index) => (
          <div
            key={index}
            className={clsx(
              styles.dot,
              subtle ? styles.subtle : styles.primary,
            )}
            style={{ animationDelay: `${index * 0.2}s` }}
          />
        ))}
      </div>
      <span className={styles.visuallyHidden}>{text}</span>
    </div>
  );
};
