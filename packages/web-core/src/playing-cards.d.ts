declare module "@letele/playing-cards" {
  import type { ComponentType, SVGProps } from "react";

  const cards: Record<
    string,
    ComponentType<
      SVGProps<SVGSVGElement> & {
        title?: string;
        titleId?: string;
      }
    >
  >;

  export = cards;
}
