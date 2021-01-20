$("#reset-data-btn").click(() => {
    let problemSelectJQ = $("#problem-select");
    let fieldSelectJQ = $("#primary-field-select");

    let problem = problemSelectJQ.val();
    let field = fieldSelectJQ.val();
    
    localStorage.clear();

    problemSelectJQ.val("none");
    problemSelectJQ.change();

    problemSelectJQ.val(problem);
    problemSelectJQ.change();

    fieldSelectJQ.val(field);
    fieldSelectJQ.change();

    objects.fieldModeService.reset();
    objects.settingsService.reset();
    objects.ctfService.reset();
    objects.clippingPlanesService.reset();
    $("#vis-mode-select").change();
});