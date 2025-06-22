export async function getHourlyScores(lat, long, date) {
    const url = `http://127.0.0.1:8000/api/score_each_hour/?lat=${lat}&long=${long}&date=${date}`;

    console.log("getting hourly scores:");
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data.scores;
    } catch (error) {
        console.log("failed");
        console.error(error);
        return null;
    }
}
