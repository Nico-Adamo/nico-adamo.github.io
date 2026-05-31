const themeDir = `${__dirname}/../../themes/hugo-eureka/`;
const siteDir = `${__dirname}/../..`;

module.exports = {
  plugins: [
    require("postcss-import")({
      path: [themeDir, siteDir],
    }),
    require("tailwindcss")(`${themeDir}assets/css/tailwind.config.js`),
    require("autoprefixer")({
      path: [themeDir],
    }),
  ],
};
