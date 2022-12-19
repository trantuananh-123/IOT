$(document).ready(function () {
    $('.selectpicker').selectpicker({});

    var x = document.getElementById("tab_type").value;
    if (x == 'predict') {
        $("#predict").addClass("active");
        $("#accurancy").removeClass("active");
        $("#train").removeClass("active");

        $("#predict_tab").addClass("active");
        $("#accurancy_tab").removeClass("active");
        $("#train_tab").removeClass("active");
    } else if (x == 'accurancy') {
        $("#accurancy").addClass("active");
        $("#predict").removeClass("active");
        $("#train").removeClass("active");

        $("#accurancy_tab").addClass("active");
        $("#predict_tab").removeClass("active");
        $("#train_tab").removeClass("active");
    } else if (x == 'train') {
        $("#train").addClass("active");
        $("#predict").removeClass("active");
        $("#accurancy").removeClass("active");

        $("#train_tab").addClass("active");
        $("#predict_tab").removeClass("active");
        $("#accurancy_tab").removeClass("active");
    }
});