// vtk.js ships its Filters/General modules without TypeScript declarations
// (the package's d.ts coverage is patchy on the General sub-tree). The
// runtime modules export factory-style objects, so we declare them as
// `any` here — call-sites cast through the imported handles explicitly.

declare module "@kitware/vtk.js/Filters/General/ImageMarchingCubes";
declare module "@kitware/vtk.js/Filters/General/WindowedSincPolyDataFilter";
