export function showJourney(name, location) {
    console.log("called");
    const encodedName = encodeURIComponent(name);
    const encodedLocation = encodeURIComponent(location);
    window.location.href = `journey.html?name=${encodedName}&location=${encodedLocation}`;    
}

window.showJourney = showJourney;