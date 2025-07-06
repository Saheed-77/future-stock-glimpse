import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export const useThemeColors = () => {
  const { theme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return {
      chartPrimary: "hsl(217, 91%, 60%)",
      chartSecondary: "hsl(142, 86%, 28%)",
      chartBackground: "hsl(210, 24%, 10%)",
      chartGrid: "hsl(210, 20%, 20%)",
      chartText: "hsl(215, 20%, 65%)",
      tooltipBg: "hsl(210, 24%, 10%)",
      tooltipText: "hsl(213, 31%, 91%)",
      tooltipBorder: "hsl(210, 20%, 20%)",
    };
  }

  const isDark = theme === "dark";

  return {
    chartPrimary: isDark ? "hsl(217, 91%, 60%)" : "hsl(217, 91%, 50%)",
    chartSecondary: isDark ? "hsl(142, 86%, 28%)" : "hsl(142, 86%, 35%)",
    chartBackground: isDark ? "hsl(210, 24%, 10%)" : "hsl(0, 0%, 100%)",
    chartGrid: isDark ? "hsl(210, 20%, 20%)" : "hsl(214, 32%, 91%)",
    chartText: isDark ? "hsl(215, 20%, 65%)" : "hsl(215, 16%, 47%)",
    tooltipBg: isDark ? "hsl(210, 24%, 10%)" : "hsl(0, 0%, 100%)",
    tooltipText: isDark ? "hsl(213, 31%, 91%)" : "hsl(222, 84%, 5%)",
    tooltipBorder: isDark ? "hsl(210, 20%, 20%)" : "hsl(214, 32%, 91%)",
  };
};