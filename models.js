export const models = [
    {
      name: 'Model A',
      contributorCount: 3,
      contributors: ['Nigeria', 'Kenya', 'South Africa'],
      contributions: {
        NG: {
          risk: 70,
          peakDischarge: '3000 m3/s',
          lastFlood: '2023-08-12',
          seasonality: 'June - September',
          stations: 5,
        },
        KE: {
          risk: 50,
          peakDischarge: '1200 m3/s',
          lastFlood: '2022-11-03',
          seasonality: 'March - May',
          stations: 3,
        },
        ZA: {
          risk: 40,
          peakDischarge: '800 m3/s',
          lastFlood: '2021-02-14',
          seasonality: 'December - February',
          stations: 4,
        },
      },
    },
    // You can define more models here
  ];